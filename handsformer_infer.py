# handsformer_infer.py
# Minimal Handsformer-style inference draft for FreiHAND
# PyTorch 1.10+ recommended

import os
import json
import math
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# ----------------------------
# Config
# ----------------------------
NUM_JOINTS = 21
NUM_VERTS  = 778
IMG_SIZE   = 224

# ----------------------------
# Simple Backbone (ResNet18)
# ----------------------------
def _conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = _conv3x3(in_ch, out_ch, stride)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_ch, out_ch)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.down  = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        idty = x
        out  = self.relu(self.bn1(self.conv1(x)))
        out  = self.bn2(self.conv2(out))
        if self.down is not None:
            idty = self.down(x)
        out += idty
        return self.relu(out)

class MiniResNet18(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256))
        self.proj   = nn.Conv2d(256, out_dim, 1)

    def forward(self, x):
        # output feature map ~ (B, C, H/32, W/32) -> project to C=out_dim
        x = self.stem(x)     # (B,64,H/4,W/4)
        x = self.layer1(x)   # (B,64)
        x = self.layer2(x)   # (B,128)
        x = self.layer3(x)   # (B,256)
        x = self.proj(x)     # (B,out_dim,h,w)
        return x

# ----------------------------
# Transformer Encoders
# (PS-DM-BERT & Perspective-BERT stand-ins)
# ----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim=256, heads=8, ff=1024, dropout=0.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ff, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

    def forward(self, x):
        # x: (B, N, C)
        return self.encoder(x)

# ----------------------------
# Handsformer Draft
# ----------------------------
class HandsformerDraft(nn.Module):
    def __init__(self, d_model=256, token_grid=7):
        super().__init__()
        self.backbone = MiniResNet18(out_dim=d_model)
        self.token_grid = token_grid  # flatten HxW tokens (e.g., 7x7=49) from feature map
        self.cls_joint = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_mesh  = nn.Parameter(torch.zeros(1, 1, d_model))

        # positional encodings for tokens
        num_tokens = token_grid * token_grid + 2  # +2 for two class tokens
        self.pos = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.01)

        # PS-DM-BERT (context modeling)
        self.psdm = TransformerBlock(dim=d_model, heads=8, ff=1024, dropout=0.0)

        # Perspective-BERT (camera-aware refinement)
        self.persp = TransformerBlock(dim=d_model, heads=8, ff=1024, dropout=0.0)

        # heads
        self.head_joints = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, NUM_JOINTS * 3)
        )
        self.head_verts = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, NUM_VERTS * 3)
        )

        # init
        nn.init.trunc_normal_(self.cls_joint, std=0.02)
        nn.init.trunc_normal_(self.cls_mesh,  std=0.02)

    def _to_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, C, H, W) -> tokens: (B, 2 + H*W, C)
        B, C, H, W = feat.shape
        assert H == self.token_grid and W == self.token_grid, \
            f"Expected {self.token_grid}x{self.token_grid} tokens, got {H}x{W}"
        grid = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        cls_joint = self.cls_joint.expand(B, -1, -1)  # (B,1,C)
        cls_mesh  = self.cls_mesh.expand(B, -1, -1)   # (B,1,C)
        tokens = torch.cat([cls_joint, cls_mesh, grid], dim=1)  # (B, 2+HW, C)
        return tokens

    def forward(self, image: torch.Tensor, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        image: (B,3,224,224), range [0,1]
        K:     (B,3,3) camera intrinsics (used for perspective encoder cue)
        returns:
          joints_3d: (B,21,3), verts_3d: (B,778,3) in (rough) metric space before scaling fix
        """
        # 1) Backbone
        feat = self.backbone(image)  # (B, C=256, h, w). Expect ~7x7 tokens
        # (ensure we get fixed token grid by adaptive pooling)
        feat = F.adaptive_avg_pool2d(feat, output_size=(self.token_grid, self.token_grid))

        # 2) Tokens + positional enc
        tokens = self._to_tokens(feat) + self.pos  # (B, N, C)

        # 3) PS-DM-BERT
        tokens = self.psdm(tokens)

        # 4) Simple camera cue: append fx, fy, cx, cy to the two class tokens via FiLM-like gating
        # (drafted idea; replace with proper perspective modeling if you extend)
        B = K.shape[0]
        fx = K[:, 0, 0].unsqueeze(-1)
        fy = K[:, 1, 1].unsqueeze(-1)
        cx = K[:, 0, 2].unsqueeze(-1)
        cy = K[:, 1, 2].unsqueeze(-1)
        cam = torch.cat([fx, fy, cx, cy], dim=-1)            # (B,4)
        cam = cam.unsqueeze(1)                               # (B,1,4)
        cam_proj = F.relu(nn.Linear(4, tokens.size(-1)).to(tokens.device)(cam))  # (B,1,C)
        tokens[:, :1, :] = tokens[:, :1, :] + cam_proj       # mod joint token
        tokens[:, 1:2, :] = tokens[:, 1:2, :] + cam_proj     # mod mesh token

        # 5) Perspective-BERT
        tokens = self.persp(tokens)

        # 6) Heads use the two class tokens
        joint_token = tokens[:, 0, :]  # (B,C)
        mesh_token  = tokens[:, 1, :]  # (B,C)
        joints = self.head_joints(joint_token).view(-1, NUM_JOINTS, 3)
        verts  = self.head_verts(mesh_token).view(-1, NUM_VERTS, 3)
        return joints, verts


# ----------------------------
# Pre/Post-processing
# ----------------------------
_preproc = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),  # [0,1]
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def _ensure_bone_scale(joints_xyz: np.ndarray, target_scale: float) -> np.ndarray:
    """
    FreiHAND defines 'scale' as ||xyz[9]-xyz[10]|| (proximal phalanx of middle finger).
    Re-scale predicted joints to match target_scale, to get metric-consistent outputs.
    """
    p1, p2 = joints_xyz[9], joints_xyz[10]
    bone = np.linalg.norm(p1 - p2) + 1e-8
    s = target_scale / bone
    return (joints_xyz - joints_xyz.mean(axis=0, keepdims=True)) * s + joints_xyz.mean(axis=0, keepdims=True)

def _rescale_verts_like_joints(verts_xyz: np.ndarray, joints_before: np.ndarray, joints_after: np.ndarray) -> np.ndarray:
    """ Apply same uniform scale used on joints to verts around their centroid. """
    c_before = joints_before.mean(axis=0)
    c_after  = joints_after.mean(axis=0)
    # scale factor is ratio of average distance to centroid
    rb = np.mean(np.linalg.norm(joints_before - c_before, axis=1)) + 1e-8
    ra = np.mean(np.linalg.norm(joints_after  - c_after,  axis=1)) + 1e-8
    s  = ra / rb
    verts_c = verts_xyz - verts_xyz.mean(axis=0, keepdims=True)
    return verts_c * s + verts_xyz.mean(axis=0, keepdims=True)

# ----------------------------
# Public API: handsformer_predict
# ----------------------------
class HandsformerRunner:
    def __init__(self, ckpt_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = HandsformerDraft(d_model=256, token_grid=7).to(self.device).eval()
        if ckpt_path and os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path, map_location=self.device)
            # Accept both full state_dict or under 'state_dict' key
            sd = state.get("state_dict", state)
            self.model.load_state_dict(sd, strict=False)
            print(f"[Handsformer] Loaded weights: {ckpt_path}")
        else:
            if ckpt_path:
                print(f"[Handsformer] WARNING: checkpoint not found at {ckpt_path}. Using random init.")

    @torch.no_grad()
    def infer(self, img_np: np.ndarray, K_np: np.ndarray, scale_scalar: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        img_np: (H,W,3) uint8/rgb
        K_np:   (3,3) intrinsics
        scale_scalar: FreiHAND provided metric bone length
        returns: xyz (21,3), verts (778,3) in meters (approx), scaled to match scale_scalar
        """
        # preprocess
        img_pil = Image.fromarray(img_np.astype(np.uint8))
        inp = _preproc(img_pil)[None, ...].to(self.device)  # (1,3,224,224)
        K   = torch.from_numpy(K_np).float()[None, ...].to(self.device)  # (1,3,3)

        # forward
        joints_t, verts_t = self.model(inp, K)  # (1,21,3), (1,778,3)
        joints = joints_t[0].cpu().numpy()
        verts  = verts_t[0].cpu().numpy()

        # center at wrist (joint 0) for stability (optional)
        wrist = joints[0].copy()
        joints -= wrist
        verts  -= wrist

        # enforce bone scale
        joints_scaled = _ensure_bone_scale(joints, float(scale_scalar))
        verts_scaled  = _rescale_verts_like_joints(verts, joints, joints_scaled)

        return joints_scaled, verts_scaled


# Singleton convenience
_runner = None

def handsformer_predict(img: np.ndarray, K: np.ndarray, scale: float,
                        ckpt_path: str = "handsformer_weights.pth") -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop-in replacement for your pred_template(img, K, scale).
    """
    global _runner
    if _runner is None:
        _runner = HandsformerRunner(ckpt_path=ckpt_path)
    xyz, verts = _runner.infer(img, K, scale)
    return xyz, verts
