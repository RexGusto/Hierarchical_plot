import os
import argparse
import pandas as pd


SORT_COLS = [
    'serial', 'dataset_name', 'model_name',
    'lr', 'seed', 'host', 'batch_size',
]


def load_process_pretraining_stats(input_file):
    # Load the two CSV files
    df_stats = pd.read_csv(input_file)

    # Add "hi-" prefix to the 'pt' column in both DataFrames
    df_stats['model_name'] = 'hi' + df_stats['pt'].astype(str)

    # Keep only the relevant columns: 'pt', 'augreg', 'acc_in1k'
    df_stats = df_stats[['model_name', 'augreg', 'acc_in1k']]
    return df_stats


def merge_pretraining_stats(df, stats_df):
    # Merge the W&B DataFrame with the pretraining stats DataFrame
    # Match 'model_name' from df with 'pt' from stats_df
    df = df.merge(stats_df, how='left', left_on='model_name', right_on='pt')
    
    # Drop the redundant 'pt' column after merging
    if 'pt' in df.columns:
        df = df.drop(columns=['pt'])

    return df


def sort_save_df(df, fp, sort_cols=['serial']):
    df = df.sort_values(by=sort_cols, ascending=[True for _ in sort_cols])
    df.to_csv(fp, header=True, index=False)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--input_file_acc', type=str,
                        default=os.path.join('data', 'hierarchical_stage1.csv'),
                        help='filename for input .csv file')
    parser.add_argument('--input_file_metrics', type=str,
                        default=os.path.join('data', 'hierarchical_feature_metrics.csv'),
                        help='filename for input .csv file')
    parser.add_argument('--input_file_pretrainings', type=str,
                        default=os.path.join('data', 'stats_pretrainings.csv'))

    # Output
    parser.add_argument('--output_file', default='hierarchical_all.csv', type=str,
                        help='File path')
    parser.add_argument('--results_dir', type=str, default='data',
                        help='The directory where results will be stored')
    parser.add_argument('--sort_cols', nargs='+', type=str, default=SORT_COLS)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    args.output_file = os.path.join(args.results_dir, args.output_file)

    # load dataframes
    df = pd.read_csv(args.input_file_acc)
    df_metrics = pd.read_csv(args.input_file_metrics)
    stats_df = load_process_pretraining_stats(args.input_file_pretrainings)
    print(len(df), df.columns, df.iloc[0])

    # merge dataframes
    df = pd.merge(df, df_metrics, how='left', on=['dataset_name', 'model_name'])
    df = pd.merge(df, stats_df, how='left', on=['model_name'])
    print(len(df), df.columns, df.iloc[0])

    # Sort and save the updated DataFrame
    sort_save_df(df, args.output_file, args.sort_cols)

    return 0

if __name__ == '__main__':
    main()