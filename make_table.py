import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from utils import filter_df, METHODS_DIC

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str, required=True)

    # Column selection
    parser.add_argument('--columns', nargs='+', type=str, required=True,
                        help='Columns to include in table')

    # Filtering (reuse yours)
    parser.add_argument('--keep_datasets', nargs='+', type=str, default=None)
    parser.add_argument('--keep_methods', nargs='+', type=str, default=None)
    parser.add_argument('--filter_datasets', nargs='+', type=str, default=None)
    parser.add_argument('--filter_methods', nargs='+', type=str, default=None)
    parser.add_argument('--keep_serials', nargs='+', type=int, default=None)
    parser.add_argument('--keep_ratios', nargs='+', type=int, default=None)
    parser.add_argument('--keep_extractor', nargs='+', type=str, default=None)

    # Output
    parser.add_argument('--output_file', type=str, default='table.png')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--fig_size', nargs='+', type=float, default=[6, 6])
    parser.add_argument('--font_size', type=int, default=8)

    # flag
    parser.add_argument('--aggregate_dataset', action='store_true',
                        help='aggregate dataset naming variants into one')

    return parser.parse_args()

def rename_methods(df, column_name='method'):
    df = df.copy()
    df[column_name] = df[column_name].map(METHODS_DIC).fillna(df[column_name])
    return df

def main():
    args = parse_args()

    df = pd.read_csv(args.input_file)

    if args.aggregate_dataset:
        base_datasets = ['aircraft', 'cub', 'cars']

        df['dataset_name'] = df['dataset_name'].apply(
            lambda x: x.split('_')[0] if x.split('_')[0] in base_datasets else x
        )

    # Reuse your filtering
    df = filter_df(
        df,
        getattr(args, 'keep_datasets', None),
        getattr(args, 'keep_methods', None),
        getattr(args, 'keep_serials', None),
        getattr(args, 'filter_datasets', None),
        getattr(args, 'filter_methods', None),
        getattr(args, 'filter_serials', None),
        getattr(args, 'keep_ratios', None),
        getattr(args, 'keep_extractor', None),
    )

    # Select only chosen columns
    df = df[args.columns]

    # Round numeric columns
    df = df.round(3)

    df = rename_methods(df, column_name='method')

    fig, ax = plt.subplots(figsize=(args.fig_size[0], args.fig_size[1]))
    ax.axis('off')

    # Prepare cell colors
    colors = []
    for _, row in df.iterrows():
        row_colors = []
        for col in df.columns:
            if col in ['abs_dif_ada', 'rel_dif_ada', 'abs_dif_acc_std', 'rel_dif_acc_std']:  # Columns to color
                val = row[col]
                if val < 0:
                    row_colors.append('green')  # light green for improvement
                elif val > 0:
                    row_colors.append("red")  # light red for worse
                else:
                    row_colors.append('white')     # neutral
            elif col in ['abs_dif_acc_mean', 'rel_dif_acc_mean']:  # Columns to color
                val = row[col]
                if val > 0:
                    row_colors.append('green')  # light green for improvement
                elif val < 0:
                    row_colors.append("red")  # light red for worse
                else:
                    row_colors.append('white')     # neutral
            else:
                row_colors.append('white')
        colors.append(row_colors)

    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellColours=colors,
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(args.font_size)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    plt.tight_layout()
    plt.savefig(args.output_file, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved table to {args.output_file}")


if __name__ == '__main__':
    main()