import argparse
import os

import numpy as np
import pandas as pd


COHORTS = {
    'brca': {
        'source_dir': 'tcga_brca',
        'target_dir': 'TCGA-BRCA',
        'source_csv': 'TCGA_BRCA_path_subtype_x10_processed.csv',
        'target_csv': 'tcga-brca_label.csv',
        'split_dir': 'BRCA_100',
        'label_map': {'IDC': 'IDC', 'ILC': 'ILC'},
    },
    'lung': {
        'source_dir': 'tcga_lung',
        'target_dir': 'TCGA-NSCLC',
        'source_csv': 'TCGA_LUNG_path_subtype_x10_processed.csv',
        'target_csv': 'tcga-nsclc_label.csv',
        'split_dir': 'NSCLC_100',
        'label_map': {'LUAD': 'LUAD', 'LUSC': 'LUSC'},
    },
    'rcc': {
        'source_dir': 'tcga_rcc',
        'target_dir': 'TCGA-RCC',
        'source_csv': 'TCGA_RCC_path_subtype_x10_processed.csv',
        'target_csv': 'tcga-kidney_label.csv',
        'split_dir': 'RCC_100',
        'label_map': {'CCRCC': 'CCRCC', 'PRCC': 'PRCC', 'CHRCC': 'CHRCC'},
    },
    'esca': {
        'source_dir': 'tcga_esca',
        'target_dir': 'TCGA-ESCA',
        'source_csv': 'TCGA_ESCA_path_subtype_x10_processed.csv',
        'target_csv': 'tcga-esca_label.csv',
        'split_dir': 'ESCA_100',
        'label_map': {
            'ESAD': 'Adenocarcinoma',
            'ESCC': 'Squamous cell carcinoma',
        },
    },
}


def force_symlink(src, dst):
    if os.path.islink(dst):
        current = os.readlink(dst)
        if current == src:
            return
        os.unlink(dst)
    elif os.path.exists(dst):
        raise FileExistsError(
            '{} already exists and is not a symlink'.format(dst))

    os.symlink(src, dst)


def write_split_csv(df, split_npz, output_csv):
    split = np.load(split_npz, allow_pickle=True)
    rows = {}
    for source_key, target_key in [
        ('train_patients', 'train'),
        ('val_patients', 'val'),
        ('test_patients', 'test'),
    ]:
        patients = set(split[source_key].tolist())
        slide_ids = df.loc[df['case_id'].isin(patients), 'slide_id'].tolist()
        rows[target_key] = slide_ids

    max_len = max(len(values) for values in rows.values())
    padded = {
        key: values + [np.nan] * (max_len - len(values))
        for key, values in rows.items()
    }
    pd.DataFrame(padded).to_csv(output_csv, index=False)
    return {key: len(values) for key, values in rows.items()}


def prepare_cohort(name, cfg, args):
    source_root = os.path.join(args.source_root, cfg['source_dir'])
    target_root = os.path.join(args.output_data_root, cfg['target_dir'])
    target_split_root = os.path.join(args.output_split_root, cfg['split_dir'])
    os.makedirs(target_root, exist_ok=True)
    os.makedirs(target_split_root, exist_ok=True)

    source_csv = os.path.join(source_root, 'table', cfg['source_csv'])
    feature_dir = os.path.join(source_root, args.feature_subdir)
    pt_dir = os.path.join(feature_dir, 'pt_files')
    pt_ids = {
        os.path.splitext(filename)[0]
        for filename in os.listdir(pt_dir)
        if filename.endswith('.pt')
    }

    df = pd.read_csv(source_csv)
    df = df[df['pathology_id'].isin(pt_ids)].copy()
    df['slide_id'] = df['pathology_id']
    df['case_id'] = df['patient_id']
    df['label'] = df['subtype'].map(cfg['label_map'])

    if df['label'].isna().any():
        missing = sorted(df.loc[df['label'].isna(), 'subtype'].unique())
        raise ValueError('Unmapped labels in {}: {}'.format(name, missing))

    target_csv = os.path.join(target_root, cfg['target_csv'])
    df[['slide_id', 'case_id', 'label']].to_csv(target_csv, index=False)
    force_symlink(feature_dir, os.path.join(target_root, args.feature_subdir))

    split_counts = []
    for fold in range(1, args.n_folds + 1):
        split_npz = os.path.join(source_root, 'datasplit', 'fold_{}.npz'.format(fold))
        output_csv = os.path.join(target_split_root, 'splits_{}.csv'.format(fold - 1))
        split_counts.append(write_split_csv(df, split_npz, output_csv))

    return {
        'cohort': name,
        'rows': len(df),
        'labels': df['label'].value_counts().to_dict(),
        'target_csv': target_csv,
        'feature_dir': feature_dir,
        'split_counts_fold0': split_counts[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_root',
                        default='/home/alan0804/CL/Dataset/03_Data/can_dataset')
    parser.add_argument('--output_data_root',
                        default='/home/alan0804/CL/Dataset/navipath_conslide')
    parser.add_argument('--output_split_root',
                        default='/home/alan0804/CL/Dataset/navipath_conslide_splits')
    parser.add_argument('--feature_subdir', default='feats-l1-s256_CONCH')
    parser.add_argument('--n_folds', type=int, default=10)
    args = parser.parse_args()

    for name, cfg in COHORTS.items():
        summary = prepare_cohort(name, cfg, args)
        print(summary)


if __name__ == '__main__':
    main()
