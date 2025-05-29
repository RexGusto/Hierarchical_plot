import os
import argparse
import numpy as np
import pandas as pd

from utils import preprocess_df, add_all_cols_group, \
    round_combine_str_mean_std, sort_df, rename_var, \
    DATASETS_DIC, METHODS_DIC


def aggregate_results_main(
    df, acc_col='val_acc_level1', serials=None, fp=None, 
    add_method_avg=True, add_dataset_avg=False,
    group_keys=['serial', 'setting', 'dataset_name', 'method']):
    # only include results from certain serials
    df = df[df['serial'].isin(serials)].copy(deep=False)


    # compute method avg (across datasets) or dataset avg (across methods)
    if add_method_avg:
        df = add_all_cols_group(df, 'dataset_name')
    if add_dataset_avg:
        df = add_all_cols_group(df, 'method')


    # compute average, stdev, max, min, ada (deviation to average ratio)
    df_std = df.groupby(group_keys, as_index=False).agg({acc_col: 'std'})
    df_max = df.groupby(group_keys, as_index=False).agg({acc_col: 'max'})
    df_min = df.groupby(group_keys, as_index=False).agg({acc_col: 'min'})
    df = df.groupby(group_keys, as_index=False).agg({acc_col: 'mean'})
    df = df.rename(columns={acc_col: 'acc_mean'})
    
    df['acc_std'] = df_std[acc_col]
    df['acc_max'] = df_max[acc_col]
    df['acc_min'] = df_min[acc_col]
    df['ada_ratio'] = 100 * (df['acc_std'] / df['acc_mean'])


    # sort dataframe
    df = sort_df(df)


    # combine mean and stdev into a single-column string (for latex/tables)
    df = round_combine_str_mean_std(df, col='acc')


    # save results
    if fp:
        df.to_csv(fp, header=True, index=False)
        print('Saved results to : ', fp)
    return df


def pivot_table(df, var='acc_mean', serial=23, fp=None, rename=True):
    # only include results from certain serials
    df = df[df['serial'] == serial].copy(deep=False)


    # pivot table so different datsaets become columns and each row is a method
    df = df.pivot(index='method', columns='dataset_name')[var]


    # sort columns based on datasets
    datasets = [ds for ds in DATASETS_DIC.keys() if ds in df.columns]
    df = df[datasets]

    # sort rows based on methods
    df['method_order'] = pd.Categorical(df.index, categories=METHODS_DIC.keys(), ordered=True)
    df = df.sort_values(by=['method_order'], ascending=True)
    df = df.drop(columns=['method_order'])


    if rename:
        df.index = df.index.map(rename_var)
        df.columns = df.columns.map(rename_var)


    if fp:
        df.to_csv(fp, header=True, index=True)
        print('Saved results to : ', fp)
    else:
        print(df)

    return 0


def summarize_results(args):
    # load dataset
    df = pd.read_csv(args.input_file)

    # preprocess to include method and setting columns,
    # drop columns
    # filter
    df = preprocess_df(
        df,
        'acc',
        getattr(args, 'keep_datasets', None),
        getattr(args, 'keep_methods', None),
        getattr(args, 'keep_serials', None),
        getattr(args, 'filter_datasets', None),
        getattr(args, 'filter_methods', None),
        getattr(args, 'filter_serials', None),
    )

    # aggregate and save results
    fp = os.path.join(args.results_dir, args.output_file)

    acc_cols = ['val_acc_level1', 'val_acc_level2', 'ap_w']
    acc_types = ['acc_mean', 'acc_max', 'ada_ratio']
    for acc_col in acc_cols:
        fn = f'{fp}_{acc_col}'
        df_main = aggregate_results_main(df, acc_col, args.main_serials, f'{fn}_main.csv')

        for serial in args.main_serials:

            for acc_type in acc_types:
                fn = f'{fp}_{acc_col}_{acc_type}_{serial}'
                df_pivoted = pivot_table(df_main, acc_type, serial, f'{fn}_pivoted.csv')

            if args.make_mean_stdev_tables:
                fn = f'{fp}_{acc_col}_{serial}'
                pivot_table(df_main, 'acc_mean_std', serial,
                            f'{fn}_pivoted_mean_std.csv',)
                pivot_table(df_main, 'acc_mean_std_latex', serial,
                            f'{fp}_{serial}_pivoted_mean_std_latex.csv',)

    return df_main


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--input_file', type=str, 
                        default=os.path.join('data', 'hierarchical_all.csv'),
                        help='filename for input .csv file from wandb')

    parser.add_argument('--keep_datasets', nargs='+', type=str, default=None)
    parser.add_argument('--keep_methods', nargs='+', type=str, default=None)
    parser.add_argument('--main_serials', nargs='+', type=int, default=[23, 24])

    # output
    parser.add_argument('--output_file', type=str, default='summary',
                        help='filename for output .csv file')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'acc'),
                        help='The directory where results will be stored')
    parser.add_argument('--make_mean_stdev_tables', action='store_true')

    args= parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    df = summarize_results(args)
    return df


if __name__ == '__main__':
    main()