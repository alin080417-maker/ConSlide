import argparse
import os

import pandas as pd
import torch


COHORTS = [
    ('NSCLC', 'TCGA-NSCLC', 'tcga-nsclc_label.csv', 'NSCLC_100'),
    ('BRCA', 'TCGA-BRCA', 'tcga-brca_label.csv', 'BRCA_100'),
    ('RCC', 'TCGA-RCC', 'tcga-kidney_label.csv', 'RCC_100'),
    ('ESCA', 'TCGA-ESCA', 'tcga-esca_label.csv', 'ESCA_100'),
]


def feature_candidates(feature_dir, slide_id, feature_format):
    if feature_format == 'h5':
        return [
            os.path.join(feature_dir, 'h5_files', '{}.h5'.format(slide_id)),
            os.path.join(feature_dir, '{}.h5'.format(slide_id)),
        ]
    return [
        os.path.join(feature_dir, 'pt_files', '{}.pt'.format(slide_id)),
        os.path.join(feature_dir, '{}.pt'.format(slide_id)),
    ]


def first_existing(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def describe_feature(path, args):
    if path.endswith('.h5'):
        import h5py

        with h5py.File(path, 'r') as handle:
            keys = list(handle.keys())
            shape = None
            if args.feature_key in handle:
                shape = tuple(handle[args.feature_key].shape)
        return 'keys={}, {} shape={}'.format(keys, args.feature_key, shape)

    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, torch.Tensor):
        return 'tensor shape={}'.format(tuple(obj.shape))
    if isinstance(obj, dict):
        keys = list(obj.keys())
        shape = None
        if args.feature_key in obj:
            value = obj[args.feature_key]
            shape = tuple(value.shape) if hasattr(value, 'shape') else None
        return 'keys={}, {} shape={}'.format(keys, args.feature_key, shape)
    return 'payload type={}'.format(type(obj))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_data_root', default='../Dataset')
    parser.add_argument('--wsi_split_root', default='../HIT/10fold_splits')
    parser.add_argument('--wsi_feature_subdir',
                        default='patch_4096/convnexts_l0l1_512_4096')
    parser.add_argument('--wsi_feature_format', default='h5',
                        choices=['h5', 'pt'])
    parser.add_argument('--feature_key', default='features')
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()

    for name, folder, csv_name, split_name in COHORTS:
        cohort_root = os.path.join(args.wsi_data_root, folder)
        csv_path = os.path.join(cohort_root, csv_name)
        feature_dir = os.path.join(cohort_root, args.wsi_feature_subdir)
        split_path = os.path.join(
            args.wsi_split_root, split_name, 'splits_{}.csv'.format(args.fold))

        print('\n[{}]'.format(name))
        print('csv: {} {}'.format(csv_path, 'OK' if os.path.exists(csv_path) else 'MISSING'))
        print('features: {} {}'.format(feature_dir, 'OK' if os.path.isdir(feature_dir) else 'MISSING'))
        print('split: {} {}'.format(split_path, 'OK' if os.path.exists(split_path) else 'MISSING'))

        if not os.path.exists(csv_path):
            continue

        slide_data = pd.read_csv(csv_path)
        if 'slide_id' not in slide_data:
            print('slide_id column missing')
            continue

        checked = 0
        missing = 0
        for slide_id in slide_data['slide_id'].dropna().head(10):
            path = first_existing(
                feature_candidates(feature_dir, slide_id, args.wsi_feature_format))
            if path is None:
                missing += 1
                continue
            if checked == 0:
                print('sample: {}'.format(path))
                print(describe_feature(path, args))
            checked += 1

        print('checked={}, missing_in_first_10={}'.format(checked, missing))


if __name__ == '__main__':
    main()
