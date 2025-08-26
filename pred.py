from __future__ import print_function, unicode_literals
import argparse
from tqdm import tqdm

from handsformer_infer import handsformer_predict
from utils.fh_utils import *


def main(base_path, pred_out_path, pred_func, set_name=None):
    """
        Main eval loop: Iterates over all evaluation samples and saves the corresponding predictions.
    """
    # default value
    if set_name is None:
        set_name = 'evaluation'

    # init output containers
    xyz_pred_list, verts_pred_list = list(), list()

    K_list = json_load(os.path.join(base_path, '%s_K.json' % set_name))
    scale_list = json_load(os.path.join(base_path, '%s_scale.json' % set_name))

    # iterate over the dataset once
    for idx in tqdm(range(db_size(set_name))):
        img = read_img(idx, base_path, set_name)

        # call predictor function
        xyz, verts = pred_func(
            img,
            np.array(K_list[idx]),
            scale_list[idx]
        )
        xyz_pred_list.append(xyz)
        verts_pred_list.append(verts)

    # dump results
    dump(pred_out_path, xyz_pred_list, verts_pred_list)


def dump(pred_out_path, xyz_pred_list, verts_pred_list):
    """ Save predictions into a json file. """
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    verts_pred_list = [x.tolist() for x in verts_pred_list]

    with open(pred_out_path, 'w') as fo:
        json.dump([xyz_pred_list, verts_pred_list], fo)

    print('Dumped %d joints and %d verts predictions to %s' %
          (len(xyz_pred_list), len(verts_pred_list), pred_out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate predictions using Handsformer.')
    parser.add_argument('base_path', type=str,
                        help='Path to where the FreiHAND dataset is located.')
    parser.add_argument('--out', type=str, default='pred.json',
                        help='File to save the predictions.')
    parser.add_argument('--ckpt', type=str, default='handsformer_weights.pth',
                        help='Path to Handsformer checkpoint.')
    args = parser.parse_args()

    # Wrap handsformer_predict so ckpt_path gets passed properly
    def pred_func(img, K, scale):
        return handsformer_predict(img, K, scale, ckpt_path=args.ckpt)

    main(
        args.base_path,
        args.out,
        pred_func=pred_func,
        set_name='evaluation'
    )
