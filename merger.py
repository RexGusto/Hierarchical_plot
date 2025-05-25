import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from math import ceil

# Set font to Times New Roman (or serif fallback)
plt.rcParams["font.family"] = 'serif'

# --- CONFIG ---
parent_folder = "to_be_merged2"
output_dir = "multi_plot_outputs"
os.makedirs(output_dir, exist_ok=True)

grid_cols = 7
img_width, img_height = 3, 2
dataset_order = ['aircraft', 'cub', 'cars']

sort_priority = {
    ('vit', 'FT'): 0,
    ('resnet', 'FT'): 1,
    ('both', 'FT'): 2,
    ('vit', 'FZ'): 3,
    ('resnet', 'FZ'): 4,
    ('both', 'FZ'): 5,
    ('both', 'FT_FZ'): 6,
}

column_labels = {
    ('vit', 'FT'): 'ViT FT',
    ('resnet', 'FT'): 'RN FT',
    ('both', 'FT'): 'ViT+RN FT',
    ('vit', 'FZ'): 'ViT FZ',
    ('resnet', 'FZ'): 'RN FZ',
    ('both', 'FZ'): 'ViT+RN FZ',
    ('both', 'FT_FZ'): 'ViT+RN FT+FZ',
}

metric_name_map = {
    'in1kacc': "ImageNet-1K Accuracy",
    'cka_avg': "Centered Kernel Alignment (CKA)",
    'dist_avg': "Euclidean Distance",
    'dist_norm_avg': "Euclidean Distance (Normalized)",
    'l2_norm': "L2 Norm",
    'cka_0': "CKA (First Layer)",
    'cka_last_layer': "CKA (Last Layer)",
    'cka_high_mean': "CKA (High Layers)",
    'cka_mid_mean': "CKA (Mid Layers)",
    'cka_low_mean': "CKA (Low Layers)",
    'msc': "Mean Silhouette Coefficient (MSC)",
    'v_intra': "Intra-Class Variance",
    's_inter': "Inter-Class Separation",
    'cis_clustering': "CIS(CLustering)",
    'cis_spectral': "CIS(Spectral)",
    'clustering_diversity': "Clustering Diversity",
    'spectral_diversity': "Spectral Diversity"
}

# --- Title Formatter ---
def infer_title_from_folder(folder_name):
    folder_name = folder_name.lower()
    for base_metric, full_title in metric_name_map.items():
        if base_metric in folder_name:
            return f"Correlations for Top-1 Accuracy vs {full_title}"
    return "Correlations for Top-1 Accuracy"

# --- Sorting Helpers ---
def extract_sort_components(filename):
    base = os.path.basename(filename).replace(".png", "")
    parts = base.split("_")
    try:
        if parts[-2] in ['FT', 'FZ']:
            method = parts[-4]
            dataset = parts[-3]
            prefix = parts[-2] + "_" + parts[-1]
        else:
            method = parts[-3]
            dataset = parts[-2]
            prefix = parts[-1]
        return dataset, method, prefix
    except IndexError:
        return "zzz", "zzz", "zzz"

def get_sort_key(filename):
    dataset, method, prefix = extract_sort_components(filename)
    dataset_idx = dataset_order.index(dataset) if dataset in dataset_order else 99
    method_prefix_idx = sort_priority.get((method, prefix), 99)
    return (dataset_idx, method_prefix_idx)

# --- Main Loop ---
for subfolder in sorted(os.listdir(parent_folder)):
    subfolder_path = os.path.join(parent_folder, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    image_files = [
        os.path.join(subfolder_path, f)
        for f in os.listdir(subfolder_path)
        if f.endswith(".png")
    ]
    if not image_files:
        continue

    image_files = sorted(image_files, key=get_sort_key)
    grid_rows = len(dataset_order)

    fig, axes = plt.subplots(
        grid_rows, grid_cols,
        figsize=(grid_cols * img_width, grid_rows * img_height)
    )
    axes = axes.flatten()

    for i, (ax, img_path) in enumerate(zip(axes, image_files)):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('off')

        col_idx = i % grid_cols
        row_idx = i // grid_cols

        # Row label (left side)
        if col_idx == 0 and row_idx < len(dataset_order):
            label_text = dataset_order[row_idx].upper() if dataset_order[row_idx] == "cub" else dataset_order[row_idx].capitalize()
            ax.text(
                -0.15, 0.5,
                label_text,
                va='center',
                ha='right',
                fontsize=10,
                transform=ax.transAxes
            )

        # Column label (bottom)
        if row_idx == len(dataset_order) - 1:
            dataset, method, prefix = extract_sort_components(img_path)
            label = column_labels.get((method, prefix), "")
            ax.text(
                0.5, -0.1,
                label,
                va='top',
                ha='center',
                fontsize=10,
                transform=ax.transAxes
            )

    # Hide any extra axes
    for ax in axes[len(image_files):]:
        ax.axis('off')

    # Title and axis labels
    fig_title = infer_title_from_folder(subfolder)
    fig.suptitle(fig_title, fontsize=12)
    # fig.text(0.04, 0.5, "Accuracy (%)", va='center', ha='left', rotation='vertical', fontsize=10)

    plt.tight_layout(pad=0.5)

    output_file = os.path.join(output_dir, f"{subfolder}_multi_plot.png")
    plt.savefig(output_file, bbox_inches='tight', dpi=400)
    plt.close()
    print(f"Saved: {output_file}")
