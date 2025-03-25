import numpy as np
import pandas as pd


SERIALS_EXPLANATIONS = [
    # main experiments
    'ft_224',
    'fz_224',
    'ft_448',
    'fz_448',

    # inference cost
    'inference_224',
    'inference_448',

    # train cost
    'train_224',
    'train_448',
]


SETTINGS_DIC = {
    'fz_224': 'IS=224',
    'fz_448': 'IS=448',
    'ft_224': 'IS=224 (FT)',
    'ft_448': 'IS=448 (FT)',
    'mtd_224': 'IS=224 (MTD)',
    'mtd_448': 'IS=448 (MTD)',
}

METHODS_VIT_FSL = [
    'hivit_base_patch16_224.orig_in21k', 'hivit_base_patch16_224_miil.in21k',
    'hideit_base_patch16_224.fb_in1k', 'hideit3_base_patch16_224.fb_in1k',
    'hideit3_base_patch16_224.fb_in22k_ft_in1k',

    'hivit_base_patch16_224.orig_in21k_fz', 'hivit_base_patch16_224_miil.in21k_fz',
    'hideit_base_patch16_224.fb_in1k_fz', 'hideit3_base_patch16_224.fb_in1k_fz',
    'hideit3_base_patch16_224.fb_in22k_ft_in1k_fz',
]

METHODS_VIT_SSL = [
    'hivit_base_patch16_clip_224.laion2b','hivit_base_patch16_224.mae', 
    'hivit_base_patch16_224.in1k_mocov3','hivit_base_patch16_224.dino',
    'hivit_base_patch16_siglip_224.v2_webli',

    'hivit_base_patch16_clip_224.laion2b_fz','hivit_base_patch16_224.mae_fz', 
    'hivit_base_patch16_224.in1k_mocov3_fz','hivit_base_patch16_224.dino_fz',
    'hivit_base_patch16_siglip_224.v2_webli_fz',
]

METHODS_RESNET_FSL = [
    'hiresnet50.tv_in1k', 'hiresnet50.tv2_in1k', 'hiresnet50.gluon_in1k', 
    'hiresnet50.fb_swsl_ig1b_ft_in1k', 'hiresnet50.fb_ssl_yfcc100m_ft_in1k',
    'hiresnet50.a1_in1k',

    'hiresnet50.tv_in1k_fz', 'hiresnet50.tv2_in1k_fz', 'hiresnet50.gluon_in1k_fz', 
    'hiresnet50.fb_swsl_ig1b_ft_in1k_fz', 'hiresnet50.fb_ssl_yfcc100m_ft_in1k_fz',
    'hiresnet50.a1_in1k_fz',
]

METHODS_RESNET_SSL = [
    'hiresnet50.in1k_byol', 'hiresnet50.in1k_mocov3', 'hiresnet50.in1k_spark', 
    'hiresnet50.in1k_supcon', 'hiresnet50.in1k_swav',
    'hiresnet50.in21k_miil',

    'hiresnet50.in1k_byol_fz', 'hiresnet50.in1k_mocov3_fz', 'hiresnet50.in1k_spark_fz', 
    'hiresnet50.in1k_supcon_fz', 'hiresnet50.in1k_swav_fz',
    'hiresnet50.in21k_miil_fz',
]

DATASETS_UFGIR = [
    'cotton',
    'soyageing',
    'soyageingr1',
    'soyageingr3',
    'soyageingr4',
    'soyageingr5',
    'soyageingr6',
    'soygene',
    'soyglobal',
    'soylocal',
]


DATASETS_DIC = {
    'aircraft': 'Aircraft',
    'cars': 'Cars',
    'cotton': 'Cotton',
    'cub': 'CUB',
    'dogs': 'Dogs',
    'dafb': 'DAFB',
    'flowers': 'Flowers',
    'food': 'Food',
    'inat17': 'iNat17',
    'moe': 'Moe',
    'nabirds': 'NABirds',
    'pets': 'Pets',
    'soyageing': 'SoyAgeing',
    'soyageingr1': 'SoyAgeingR1',
    'soyageingr3': 'SoyAgeingR3',
    'soyageingr4': 'SoyAgeingR4',
    'soyageingr5': 'SoyAgeingR5',
    'soyageingr6': 'SoyAgeingR6',
    'soygene': 'SoyGene',
    'soyglobal': 'SoyGlobal',
    'soylocal': 'SoyLocal',
    'vegfru': 'VegFru',
    'all': 'Average',
}

METHODS_DIC = {
    # Resnet FSL Models
    'hiresnet50.tv_in1k': 'Torchv1', 
    'hiresnet50.tv2_in1k': 'Torchv2', 
    'hiresnet50.gluon_in1k': 'Gluon', 
    'hiresnet50.fb_swsl_ig1b_ft_in1k': 'SWSL',
    'hiresnet50.fb_ssl_yfcc100m_ft_in1k': 'SSL',
    'hiresnet50.a1_in1k': 'A1',

    'hiresnet50.tv_in1k_fz': 'Torchv1', 
    'hiresnet50.tv2_in1k_fz': 'Torchv2', 
    'hiresnet50.gluon_in1k_fz': 'Gluon', 
    'hiresnet50.fb_swsl_ig1b_ft_in1k_fz': 'SWSL',
    'hiresnet50.fb_ssl_yfcc100m_ft_in1k_fz': 'SSL',
    'hiresnet50.a1_in1k_fz': 'A1',

    # Resnet SSL Models
    'hiresnet50.in1k_byol': 'BYOL', 
    'hiresnet50.in1k_mocov3': 'MoCo v3', 
    'hiresnet50.in1k_spark': 'SParK', 
    'hiresnet50.in1k_supcon': 'Supcon', 
    'hiresnet50.in1k_swav': 'SwAV',
    'hiresnet50.in21k_miil': 'MIIL',

    'hiresnet50.in1k_byol_fz': 'BYOL', 
    'hiresnet50.in1k_mocov3_fz': 'MoCo v3', 
    'hiresnet50.in1k_spark_fz': 'SParK', 
    'hiresnet50.in1k_supcon_fz': 'Supcon', 
    'hiresnet50.in1k_swav_fz': 'SwAV',
    'hiresnet50.in21k_miil_fz': 'MIIL',

    # ViT FSL Models
    'hivit_base_patch16_224.orig_in21k': 'ViT',
    'hideit_base_patch16_224.fb_in1k': 'DeiT',
    'hivit_base_patch16_224_miil.in21k': 'IN21k-P',
    'hideit3_base_patch16_224.fb_in22k_ft_in1k': 'DeiT 3 (IN21k)',
    'hideit3_base_patch16_224.fb_in1k': 'DeiT 3 (IN1k)',

    'hivit_base_patch16_224.orig_in21k_fz': 'ViT',
    'hideit_base_patch16_224.fb_in1k_fz': 'DeiT',
    'hivit_base_patch16_224_miil.in21k_fz': 'IN21k-P',
    'hideit3_base_patch16_224.fb_in22k_ft_in1k_fz': 'DeiT 3 (IN21k)',
    'hideit3_base_patch16_224.fb_in1k_fz': 'DeiT 3 (IN1k)',

    # ViT SSL Models
    'hivit_base_patch16_224.in1k_mocov3': 'MoCo v3',
    'hivit_base_patch16_224.dino': 'DINO',
    'hivit_base_patch16_224.mae': 'MAE',
    'hivit_base_patch16_clip_224.laion2b': 'CLIP',
    'hivit_base_patch16_siglip_224.v2_webli': 'SigLIP',

    'hivit_base_patch16_224.in1k_mocov3_fz': 'MoCo v3',
    'hivit_base_patch16_224.dino_fz': 'DINO',
    'hivit_base_patch16_224.mae_fz': 'MAE',
    'hivit_base_patch16_clip_224.laion2b_fz': 'CLIP',
    'hivit_base_patch16_siglip_224.v2_webli_fz': 'SigLIP',
}


VAR_DIC = {
    'setting': 'Setting',
    'acc': 'Accuracy (%)',
    'acc_std': 'Accuracy Std. Dev. (%)',
    'dataset_name': 'Dataset',
    'method': 'Method',
    'family': 'Method Family',
    'flops': 'Inference FLOPs (10^9)',
    'time_train': 'Train Time (hours)',
    'vram_train': 'Train VRAM (GB)',
    'trainable_percent': 'Task-Specific Parameters (%)',
    'no_params': 'Number of Parameters (10^6)',
    'no_params_trainable': 'Task-Specific Parameters (10^6)',
    'no_params_total': 'Total Parameters (10^6)',
    'no_params_trainable_total': 'Total Task-Specific Params. (10^6)',
    'flops_inference': 'Inference FLOPs (10^9)',
    'tp_stream': 'Stream Throughput (Images/s)',
    'vram_stream': 'Stream VRAM (GB)',
    'latency_stream': 'Stream Latency (s)',
    'tp_batched': 'Batched Throughput  (Images/s)',
    'vram_batched': 'Batched VRAM (GB)',
}


def rename_var(x):
    if x in SETTINGS_DIC.keys():
        return SETTINGS_DIC[x]
    elif x in METHODS_DIC.keys():
        return METHODS_DIC[x]
    elif x in DATASETS_DIC.keys():
        return DATASETS_DIC[x]
    elif x in VAR_DIC.keys():
        return VAR_DIC[x]
    return x


def rename_vars(df, var_rename=False, args=None):
    if 'setting' in df.columns:
        df['setting'] = df['setting'].apply(rename_var)
    if 'method' in df.columns:
        df['method'] = df['method'].apply(rename_var)
    if 'dataset_name' in df.columns:
        df['dataset_name'] = df['dataset_name'].apply(rename_var)
    if 'family' in df.columns:
        df['family'] = df['family'].apply(rename_var)

    if var_rename:
        df.rename(columns=VAR_DIC, inplace=True)
        for k, v in VAR_DIC.items():
            if k == args.x_var_name:
                args.x_var_name = v
            elif k == args.y_var_name:
                args.y_var_name = v
            elif k == args.hue_var_name:
                args.hue_var_name = v
            elif k == args.style_var_name:
                args.style_var_name = v
            elif k == args.size_var_name:
                args.size_var_name = v

    return df


def add_setting(df):
    conditions = [
        ((df['serial'] == 1) & (df['freeze_backbone'] == False)),
        ((df['serial'] == 1) & (df['freeze_backbone'] == True)),
        ((df['serial'] == 3) & (df['freeze_backbone'] == False)),
        ((df['serial'] == 3) & (df['freeze_backbone'] == True)),

        (df['serial'] == 40),
        (df['serial'] == 41),

        (df['serial'] == 42),
        (df['serial'] == 43),
    ]

    df['setting'] = np.select(conditions, SERIALS_EXPLANATIONS, default='')
    return df


def load_df(input_file):
    df = pd.read_csv(input_file)

    # methods
    df = df.fillna({'classifier': '', 'selector': ''})

    df['freeze_backbone_str'] = df['freeze_backbone'].apply(lambda x: '_fz' if x is True else '')

    df['method'] = df['model_name'] + df['freeze_backbone_str']

    df = add_setting(df)

    df.rename(columns={'val_acc_species': 'acc'}, inplace=True)
    return df


def keep_columns(df, type='acc'):
    if type == 'acc':
        keep = ['acc', 'dataset_name', 'serial', 'setting', 'method', 'lr']
    elif type == 'inference_cost':
        keep = ['host', 'serial', 'setting', 'method', 'batch_size', 'throughput',
                'flops', 'max_memory']
    elif type == 'train_cost':
        keep = ['host', 'serial', 'setting', 'method', 'batch_size', 'epochs',
                'dataset_name', 'num_images_train', 'num_images_val',
                'time_total', 'flops', 'max_memory',
                'no_params_trainable', 'no_params']

    df = df[keep]
    return df


def filter_df(df, keep_datasets=None, keep_methods=None, keep_serials=None,
              filter_datasets=None, filter_methods=None, filter_serials=None):
    
    if keep_datasets:
        df = df[df['dataset_name'].isin(keep_datasets)]

    if keep_methods:
        df = df[df['method'].isin(keep_methods)]

    if keep_serials:
        df = df[df['serial'].isin(keep_serials)]

    if filter_datasets:
        df = df[~df['dataset_name'].isin(filter_datasets)]

    if filter_methods:
        df = df[~df['method'].isin(filter_methods)]

    if filter_serials:
        df = df[~df['serial'].isin(filter_serials)]

    return df


def preprocess_df(
    input_file, type='acc', keep_datasets=None, keep_methods=None, keep_serials=None,
    filter_datasets=None, filter_methods=None, filter_serials=None):
    # load dataset and preprocess to include method and setting columns, rename val_acc to acc
    df = load_df(input_file)
    # print(df['acc'])

    # drop columns
    df = keep_columns(df, type=type)

    # filter
    df = filter_df(df, keep_datasets, keep_methods, keep_serials,
                   filter_datasets, filter_methods, filter_serials)
    
    df = sort_df(df)
    # print(df)
    return df


def add_all_cols_group(df, col='dataset_name'):
    subset = df.copy(deep=False)
    subset[col] = 'all'
    df = pd.concat([df, subset], axis=0, ignore_index=True)
    return df


def drop_na(df, args):
    subset = [args.x_var_name, args.y_var_name]
    if args.hue_var_name:
        subset.append(args.hue_var_name)
    if args.style_var_name:
        subset.append(args.style_var_name)
    if args.size_var_name:
        subset.append(args.size_var_name)
    df = df.dropna(subset=subset)
    return df


def sort_df(df, method_only=False, raw_data=False):
    if raw_data:
        df['dataset_order'] = pd.Categorical(df['dataset_name'], categories=DATASETS_DIC.keys(), ordered=True)
        df['model_name_order'] = pd.Categorical(df['model_name'], categories=METHODS_DIC.keys(), ordered=True)

        df = df.sort_values(by=['serial', 'dataset_name', 'method_order'], ascending=True)
        df = df.drop(columns=['model_name_order', 'dataset_order'])
    elif method_only:
        df['method_order'] = pd.Categorical(df['method'], categories=METHODS_DIC.keys(), ordered=True)

        df = df.sort_values(by=['serial', 'setting', 'method_order'], ascending=True)
        df = df.drop(columns=['method_order'])
    else:
        df['dataset_order'] = pd.Categorical(df['dataset_name'], categories=DATASETS_DIC.keys(), ordered=True)
        df['method_order'] = pd.Categorical(df['method'], categories=METHODS_DIC.keys(), ordered=True)

        df = df.sort_values(by=['serial', 'setting', 'dataset_order', 'method_order'], ascending=True)
        df = df.drop(columns=['method_order', 'dataset_order'])
    return df



def round_combine_str_mean_std(df, col='acc'):
    df[f'{col}'] = df[f'{col}'].round(2)
    df[f'{col}_std'] = df[f'{col}_std'].round(2)

    df[f'{col}_mean_std_latex'] = df[f'{col}'].astype(str) + "\\pm{" + df[f'{col}_std'].astype(str) + "}"
    df[f'{col}_mean_std'] = df[f'{col}'].astype(str) + "+-" + df[f'{col}_std'].astype(str)

    return df


def group_by_family(x):
    classifiers = ('vit_b16_cls', 'vit_b16_lrblp', 'vit_b16_mpncov', 'vit_b16_ifacls')
    fgir = ('vit_b16_cls_psm', 'vit_b16_cls_maws', 'vit_b16_cal',
              'vit_b16_avg_cls_rollout', 'vit_b16_cls_glsim',)
    ufgir =  ('clevit', 'csdnet', 'mixvit', 'vit_b16_sil')
    pefgir = ['vit_b16_cls_fz']
    if x in fgir:
        return 'fgir'
    elif x in ufgir:
        return 'ufgir'
    elif x in pefgir:
        return 'pefgir'
    elif x in classifiers:
        return 'classifier'
    return x

