# !/bin/bash

# # download
# python download_save_wandb_data.py --serials 23 24 --output_file hierarchical_stage1.csv
# python download_save_wandb_data_feature_metrics.py

# # merge data from acc, corr metrics and pretraining stats
# python merge_acc_metrics_stats.py

# # lr script
# # python lr_script.py

# # # accuracy
# python summarize_acc.py
# python summarize_acc_metrics.py

# # cost metrics including flops, no params, trainable params
# python summarize_cost.py
# python summarize_cost.py --acc_to_use val_acc_level2 --output_file cost_val_acc_level2
# python summarize_cost.py --acc_to_use ap_w --output_file cost_ap_w

# Defining datasets, serials, and model arrays
datasets_array=('aircraft' 'cub' 'cars')
serials=('23' '24')

all_resnet=('hiresnet50.tv_in1k' 'hiresnet50.tv2_in1k' 'hiresnet50.gluon_in1k' 'hiresnet50.fb_swsl_ig1b_ft_in1k' 'hiresnet50.fb_ssl_yfcc100m_ft_in1k' 'hiresnet50.a1_in1k' 'hiresnet50.in1k_mocov3' 'hiresnet50.in1k_spark' 'hiresnet50.in1k_supcon' 'hiresnet50.in1k_swav' 'hiresnet50.in21k_miil' 'hiresnet50.tv_in1k_fz' 'hiresnet50.tv2_in1k_fz' 'hiresnet50.gluon_in1k_fz' 'hiresnet50.fb_swsl_ig1b_ft_in1k_fz' 'hiresnet50.fb_ssl_yfcc100m_ft_in1k_fz' 'hiresnet50.a1_in1k_fz' 'hiresnet50.in1k_mocov3_fz' 'hiresnet50.in1k_spark_fz' 'hiresnet50.in1k_supcon_fz' 'hiresnet50.in1k_swav_fz' 'hiresnet50.in21k_miil_fz')
all_vit=('hivit_base_patch16_224.orig_in21k' 'hivit_base_patch16_224_miil.in21k' 'hideit_base_patch16_224.fb_in1k' 'hideit3_base_patch16_224.fb_in1k' 'hideit3_base_patch16_224.fb_in22k_ft_in1k' 'hivit_base_patch16_clip_224.laion2b' 'hivit_base_patch16_224.mae' 'hivit_base_patch16_224.in1k_mocov3' 'hivit_base_patch16_224.dino' 'hivit_base_patch16_siglip_224.v2_webli' 'hivit_base_patch16_224.orig_in21k_fz' 'hivit_base_patch16_224_miil.in21k_fz' 'hideit_base_patch16_224.fb_in1k_fz' 'hideit3_base_patch16_224.fb_in1k_fz' 'hideit3_base_patch16_224.fb_in22k_ft_in1k_fz' 'hivit_base_patch16_clip_224.laion2b_fz' 'hivit_base_patch16_224.mae_fz' 'hivit_base_patch16_224.in1k_mocov3_fz' 'hivit_base_patch16_224.dino_fz' 'hivit_base_patch16_siglip_224.v2_webli_fz')

all_resnet_nofz=('hiresnet50.tv_in1k' 'hiresnet50.tv2_in1k' 'hiresnet50.gluon_in1k' 'hiresnet50.fb_swsl_ig1b_ft_in1k' 'hiresnet50.fb_ssl_yfcc100m_ft_in1k' 'hiresnet50.a1_in1k' 'hiresnet50.in1k_mocov3' 'hiresnet50.in1k_spark' 'hiresnet50.in1k_supcon' 'hiresnet50.in1k_swav' 'hiresnet50.in21k_miil')
all_vit_nofz=('hivit_base_patch16_224.orig_in21k' 'hivit_base_patch16_224_miil.in21k' 'hideit_base_patch16_224.fb_in1k' 'hideit3_base_patch16_224.fb_in1k' 'hideit3_base_patch16_224.fb_in22k_ft_in1k' 'hivit_base_patch16_clip_224.laion2b' 'hivit_base_patch16_224.mae' 'hivit_base_patch16_224.in1k_mocov3' 'hivit_base_patch16_224.dino' 'hivit_base_patch16_siglip_224.v2_webli')

all_resnet_onlyfz=('hiresnet50.tv_in1k_fz' 'hiresnet50.tv2_in1k_fz' 'hiresnet50.gluon_in1k_fz' 'hiresnet50.fb_swsl_ig1b_ft_in1k_fz' 'hiresnet50.fb_ssl_yfcc100m_ft_in1k_fz' 'hiresnet50.a1_in1k_fz' 'hiresnet50.in1k_mocov3_fz' 'hiresnet50.in1k_spark_fz' 'hiresnet50.in1k_supcon_fz' 'hiresnet50.in1k_swav_fz' 'hiresnet50.in21k_miil_fz')
all_vit_onlyfz=('hivit_base_patch16_224.orig_in21k_fz' 'hivit_base_patch16_224_miil.in21k_fz' 'hideit_base_patch16_224.fb_in1k_fz' 'hideit3_base_patch16_224.fb_in1k_fz' 'hideit3_base_patch16_224.fb_in22k_ft_in1k_fz' 'hivit_base_patch16_clip_224.laion2b_fz' 'hivit_base_patch16_224.mae_fz' 'hivit_base_patch16_224.in1k_mocov3_fz' 'hivit_base_patch16_224.dino_fz' 'hivit_base_patch16_siglip_224.v2_webli_fz')

all_models=("${all_resnet[@]}" "${all_vit[@]}")
all_models_ft=("${all_resnet_nofz[@]}" "${all_vit_nofz[@]}")
all_models_fz=("${all_resnet_onlyfz[@]}" "${all_vit_onlyfz[@]}")

resnet_fsl=('hiresnet50.tv_in1k' 'hiresnet50.tv2_in1k' 'hiresnet50.gluon_in1k' 'hiresnet50.a1_in1k' 'hiresnet50.in21k_miil' 'hiresnet50.tv_in1k_fz' 'hiresnet50.tv2_in1k_fz' 'hiresnet50.gluon_in1k_fz' 'hiresnet50.a1_in1k_fz' 'hiresnet50.in21k_miil_fz')
resnet_semisl=('hiresnet50.fb_swsl_ig1b_ft_in1k' 'hiresnet50.fb_ssl_yfcc100m_ft_in1k' 'hiresnet50.fb_swsl_ig1b_ft_in1k_fz' 'hiresnet50.fb_ssl_yfcc100m_ft_in1k_fz')
resnet_generative=('hiresnet50.in1k_spark' 'hiresnet50.in1k_spark_fz')
resnet_discrim=('hiresnet50.in1k_mocov3' 'hiresnet50.in1k_supcon' 'hiresnet50.in1k_swav' 'hiresnet50.in1k_mocov3_fz' 'hiresnet50.in1k_supcon_fz' 'hiresnet50.in1k_swav_fz')

vit_fsl=('hivit_base_patch16_224.orig_in21k' 'hivit_base_patch16_224_miil.in21k' 'hideit_base_patch16_224.fb_in1k' 'hideit3_base_patch16_224.fb_in1k' 'hideit3_base_patch16_224.fb_in22k_ft_in1k' 'hivit_base_patch16_224.orig_in21k_fz' 'hivit_base_patch16_224_miil.in21k_fz' 'hideit_base_patch16_224.fb_in1k_fz' 'hideit3_base_patch16_224.fb_in1k_fz' 'hideit3_base_patch16_224.fb_in22k_ft_in1k_fz')
vit_generative=('hivit_base_patch16_224.mae' 'hivit_base_patch16_224.mae_fz')
vit_discrim=('hivit_base_patch16_clip_224.laion2b' 'hivit_base_patch16_224.in1k_mocov3' 'hivit_base_patch16_224.dino' 'hivit_base_patch16_siglip_224.v2_webli' 'hivit_base_patch16_clip_224.laion2b_fz' 'hivit_base_patch16_224.in1k_mocov3_fz' 'hivit_base_patch16_224.dino_fz' 'hivit_base_patch16_siglip_224.v2_webli_fz')

# Updated accuracies array
y_ax=('acc_max' 'ada_ratio')
accuracies=('val_acc_level1' 'val_acc_level2' 'ap_w')
methods=('resnet' 'vit')
# Added CKA last layer metrics to the list
metrics=('acc_in1k' 'MSC_train' 'MSC_test' 'V_intra_train' 'V_intra_test' 'S_inter_train' 'S_inter_test'
         'cis_clustering_diversity_train' 'cis_clustering_diversity_test' 'cis_spectral_diversity_train' 'cis_spectral_diversity_test'
         'cka_avg_train' 'cka_avg_test' 'dist_avg_train' 'dist_avg_test' 'dist_norm_avg_train'
         'dist_norm_avg_test' 'l2_norm_avg_train' 'l2_norm_avg_test'
         'cka_0_train' 'cka_0_test' 'cka_high_mean_train' 'cka_mid_mean_train' 'cka_low_mean_train'
         'cka_high_mean_test' 'cka_mid_mean_test' 'cka_low_mean_test' 'clustering_diversity_train' 'spectral_diversity_train'
         'cka_last_layer_train' 'cka_last_layer_test'
         # New metrics
         'cis_cka_0_train' 'cis_cka_0_test' 'cis_cka_last_train' 'cis_cka_last_test'
         'cis_dist_0_train' 'cis_dist_0_test' 'cis_dist_last_train' 'cis_dist_last_test')

#Train only
metrics=('acc_in1k' 'MSC_train' 'V_intra_train' 'S_inter_train' 
         'cis_clustering_diversity_train' 'cis_spectral_diversity_train' 
         'cka_avg_train' 'dist_avg_train' 'dist_norm_avg_train' 
         'l2_norm_avg_train' 'cka_0_train' 'cka_high_mean_train' 
         'cka_mid_mean_train' 'cka_low_mean_train' 'clustering_diversity_train' 
         'spectral_diversity_train' 'cka_last_layer_train' 
         # New metrics
         'cis_cka_0_train' 'cis_cka_last_train' 'cis_dist_0_train' 
         'cis_dist_last_train')

# # Generating plots for each accuracy metric and y_ax
# for accuracy in "${accuracies[@]}"; do
#     # Setting input path based on accuracy
#     input_path="results_all/acc_metrics/summary_${accuracy}_all.csv"
    
#     for y_var in "${y_ax[@]}"; do
#         for metric in "${metrics[@]}"; do
#             # Determine if this is a CKA last layer metric and set state
#             if [[ "$metric" == "cka_last_layer_train" ]]; then
#                 state="train"
#                 # Use acc folder for CKA last layer plots
#                 base_cmd="python plot.py --type_plot reg --fig_size 6.5 4 --summarized --results_dir results_all/new_plots/${accuracy} --font_scale 3.2 --font_size_correlations 34 --dpi 160"
#             elif [[ "$metric" == "cka_last_layer_test" ]]; then
#                 state="test"
#                 # Use acc folder for CKA last layer plots
#                 base_cmd="python plot.py --type_plot reg --fig_size 6.5 4 --summarized --results_dir results_all/new_plots/${accuracy} --font_scale 3.2 --font_size_correlations 34 --dpi 160"
#             else
#                 state=""
#                 base_cmd="python plot.py --type_plot reg --fig_size 6.5 4 --summarized --results_dir results_all/new_plots/${accuracy} --font_scale 3.2 --font_size_correlations 34 --dpi 160"
#             fi

#             # Plots for ViT+FT, ViT+FZ, RN+FT, RN+FZ
#             for dataset in "${datasets_array[@]}"; do
#                 for serial in "${serials[@]}"; do
#                     for method in "${methods[@]}"; do
#                         if [[ "$method" == "resnet" ]]; then
#                             model_used=${all_resnet[*]}
#                             model_label="rn"
#                             # Use specific CKA layer for ResNet (16 layers)
#                             if [[ "$metric" == "cka_last_layer_train" ]]; then
#                                 x_var="cka_15_train"
#                             elif [[ "$metric" == "cka_last_layer_test" ]]; then
#                                 x_var="cka_15_test"
#                             else
#                                 x_var="${metric}"
#                             fi
#                         else
#                             model_used=${all_vit[*]}
#                             model_label="vit"
#                             # Use specific CKA layer for ViT (12 layers)
#                             if [[ "$metric" == "cka_last_layer_train" ]]; then
#                                 x_var="cka_11_train"
#                             elif [[ "$metric" == "cka_last_layer_test" ]]; then
#                                 x_var="cka_11_test"
#                             else
#                                 x_var="${metric}"
#                             fi
#                         fi

#                         if [[ "$serial" -eq 24 ]]; then
#                             prefix="fz"
#                             if [[ "$metric" == "cka_last_layer_train" || "$metric" == "cka_last_layer_test" ]]; then
#                                 add_cmd="--y_var_name ${y_var} --x_var_name ${x_var}"
#                             else
#                                 add_cmd="--y_var_name ${y_var} --x_var_name ${x_var}"
#                             fi
#                         else
#                             prefix="ft"
#                             if [[ "$metric" == "cka_last_layer_train" || "$metric" == "cka_last_layer_test" ]]; then
#                                 add_cmd="--y_var_name ${y_var} --x_var_name ${x_var}_ft"
#                             elif [[ "$metric" == "acc_in1k" || "$metric" == "ada_ratio" ]]; then
#                                 add_cmd="--y_var_name ${y_var} --x_var_name ${metric}"
#                             else
#                                 add_cmd="--y_var_name ${y_var} --x_var_name ${x_var}_ft"
#                             fi
#                         fi

#                         # Updated output file name format: {settings}_{y_var}_{metric}_{dataset}
#                         output_file="${model_label}${prefix}_${y_var}_${metric,,}_${dataset}"
#                         cmd="${base_cmd} --input_file ${input_path} ${add_cmd} --title '' --keep_datasets ${dataset} --keep_serials ${serial} --keep_methods ${model_used[*]} --output_file ${output_file}"
#                         echo ""
#                         echo "Running: ${cmd}"
#                         eval "${cmd}"
#                     done
#                 done
#             done

#             # Plots for ViT+RN+FT, ViT+RN+FZ
#             # y_var vs metrics_ft & y_var vs metric_fz
#             for dataset in "${datasets_array[@]}"; do
#                 for serial in "${serials[@]}"; do
#                     if [[ "$serial" -eq 24 ]]; then
#                         model_used=${all_models_fz[*]}
#                         prefix="fz"
#                         if [[ "$metric" == "cka_last_layer_train" || "$metric" == "cka_last_layer_test" ]]; then
#                             add_cmd="--y_var_name ${y_var} --x_var_name cka_last_${state}"
#                         else
#                             add_cmd="--y_var_name ${y_var} --x_var_name ${metric}"
#                         fi
#                     else
#                         model_used=${all_models_ft[*]}
#                         prefix="ft"
#                         if [[ "$metric" == "cka_last_layer_train" || "$metric" == "cka_last_layer_test" ]]; then
#                             add_cmd="--y_var_name ${y_var} --x_var_name cka_last_${state}_ft"
#                         elif [[ "$metric" == "acc_in1k" || "$metric" == "ada_ratio" ]]; then
#                             add_cmd="--y_var_name ${y_var} --x_var_name ${metric}"
#                         else
#                             add_cmd="--y_var_name ${y_var} --x_var_name ${metric}_ft"
#                         fi
#                     fi
#                     # Updated output file name format
#                     output_file="both${prefix}_${y_var}_${metric,,}_${dataset}"
#                     cmd="${base_cmd} --input_file ${input_path} ${add_cmd} --title '' --keep_datasets ${dataset} --keep_serials ${serial} --keep_methods ${model_used[*]} --output_file ${output_file}"
#                     echo ""
#                     echo "Running: ${cmd}"
#                     eval "${cmd}"
#                 done
#             done

#             # New plot: y_var_fz vs metric_ft and y_var_ft vs metric_fz
#             for dataset in "${datasets_array[@]}"; do
#                 for serial in "${serials[@]}"; do
#                     if [[ "$serial" -eq 24 ]]; then
#                         # model_used=${all_models_fz[*]}
#                         # if [[ "$metric" == "cka_last_layer_train" || "$metric" == "cka_last_layer_test" ]]; then
#                         #     add_cmd="--y_var_name ${y_var} --x_var_name cka_last_${state}_matched_ft"
#                         #     alter_name="bothaccfzvsmetricft"
#                         # elif [[ "$metric" == "acc_in1k" || "$metric" == "ada_ratio" ]]; then
#                         #     add_cmd="--y_var_name ${y_var} --x_var_name ${metric}"
#                         #     alter_name="bothaccfzvsmetricft"
#                         # else
#                         #     add_cmd="--y_var_name ${y_var} --x_var_name ${metric}_ft"
#                         #     alter_name="bothaccfzvsmetricft"
#                         # fi
#                         continue
#                     fi
#                     # else
#                     model_used=${all_models_ft[*]}
#                     if [[ "$metric" == "cka_last_layer_train" || "$metric" == "cka_last_layer_test" ]]; then
#                         add_cmd="--y_var_name ${y_var} --x_var_name cka_last_${state}_matched"
#                         alter_name="bothaccftvsmetricfz"
#                     else
#                         add_cmd="--y_var_name ${y_var} --x_var_name ${metric}"
#                         alter_name="bothaccftvsmetricfz"
#                     fi
#                     # fi
#                     # Updated output file name format
#                     output_file="${alter_name}_${y_var}_${metric,,}_${dataset}"
#                     cmd="${base_cmd} --input_file ${input_path} ${add_cmd} --title '' --keep_datasets ${dataset} --keep_serials ${serial} --keep_methods ${model_used[*]} --output_file ${output_file}"
#                     echo ""
#                     echo "Running: ${cmd}"
#                     eval "${cmd}"
#                 done
#             done

#             # Plots for ViT+RN+FT+FZ 
#             for dataset in "${datasets_array[@]}"; do
#                 # Updated output file name format
#                 output_file="bothft+fz_${y_var}_${metric,,}_${dataset}"
#                 if [[ "$metric" == "cka_last_layer_train" || "$metric" == "cka_last_layer_test" ]]; then
#                     add_cmd="--y_var_name ${y_var} --x_var_name cka_last_${state}_matched"
#                 elif [[ "$metric" == "acc_in1k" || "$metric" == "ada_ratio" ]]; then
#                     add_cmd="--y_var_name ${y_var} --x_var_name ${metric}"
#                 else
#                     add_cmd="--y_var_name ${y_var} --x_var_name ${metric}_matched"
#                 fi
#                 cmd="${base_cmd} --input_file ${input_path} ${add_cmd} --title '' --keep_datasets ${dataset} --keep_serials ${serials[*]} --keep_methods ${all_models[*]} --output_file ${output_file}"
#                 echo ""
#                 echo "Running: ${cmd}"
#                 eval "${cmd}"
#             done
#         done
#     done
# done
accuracies=('ap_w')
base_cmd="python merge_corr_plots.py"
for accuracy in "${accuracies[@]}"; do
    if [[ "$accuracy" == "val_acc_level1" ]]; then
        name="acc1_multiplot"
        title_extend="Level 1 Accuracy"
    elif [[ "$accuracy" == "val_acc_level2" ]]; then
        name="acc2_multiplot"
        title_extend="Level 2 Accuracy"
    else
        name="wap_multiplot"
        title_extend="Weighted Accuracy Precision"
    fi
    cmd="${base_cmd} --input_folder results_all/new_plots/${accuracy} --output_folder results_all/new_plots/merged/${name} --output_file ${name} --title '${title_extend}' --x_filter rnfz rnft vitfz vitft"
    echo ""
    echo "Running: ${cmd}"
    eval "${cmd}"
done

# top3_best_models=('hiresnet50.in1k_mocov3' 'hiresnet50.in1k_swav' 'hiresnet50.in1k_spark' 'hiresnet50.a1_in1k' 'hivit_base_patch16_224.orig_in21k' 'hivit_base_patch16_siglip_224.v2_webli' 'hivit_base_patch16_224.dino' 'hivit_base_patch16_clip_224.laion2b')
# # ViT best vs resnet best top-1
# base_cmd="python plot.py --y_label 'Accuracy(%)' --y_var_name val_acc_level1 --hue_var_name dataset_name --x_rotation 20 --x_var_name method --type_plot box --loc_legend 'lower left' --hue_var_name dataset_name --fig_size 12 8 --results_dir results_all/new_plots/all_model_performance"
# output_file="top3_best_models_plot_top1"
# cmd="${base_cmd} --input_file data/hierarchical_all.csv --font_scale 1.75 --keep_serials 23 24 --keep_methods ${top3_best_models[*]} --output_file ${output_file} --title 'Top-2 Best & Worst-2 for Both Architecture (Top-1 Acc)' --keep_datasets ${datasets_array[*]}"
# echo "Running: ${cmd}"
# eval "${cmd}"

# # ViT best vs resnet best ap_w
# base_cmd="python plot.py --y_label 'Weighted Average Precision(wAP, %)' --y_var_name ap_w --hue_var_name dataset_name --x_rotation 20 --x_var_name method --type_plot box --loc_legend 'lower left' --hue_var_name dataset_name --fig_size 12 8 --results_dir results_all/new_plots/all_model_performance"
# output_file="top3_best_models_plot_ap_w"
# cmd="${base_cmd} --input_file data/hierarchical_all.csv --font_scale 1.75 --keep_serials 23 24 --keep_methods ${top3_best_models[*]} --output_file ${output_file} --title 'Top-2 Best & Worst-2 for Both Architecture (wAP)' --keep_datasets ${datasets_array[*]}"
# echo "Running: ${cmd}"
# eval "${cmd}"

# # Difference of maximum accuracy per model top-1
# base_cmd="python plot.py --y_label 'Accuracy(%)' --font_scale 1.75 --y_var_name acc_max --type_plot box --fig_size 10 6 --summarized --results_dir results_all/new_plots/dif_max_acc"
# output_file="max_models_performance_top1"
# cmd="${base_cmd} --input_file results_all/acc/summary_val_acc_level1_main.csv --keep_serials ${serials[*]} --x_label 'Datasets' --x_var_name dataset_name --hue_var_name serial --keep_methods ${all_resnet[*]} --keep_datasets ${datasets_array[*]} --output_file ${output_file} --title 'Max Accuracies Variations Between Resnet Models (Top1)'"
# echo "Running: ${cmd}"
# eval "${cmd}"

# base_cmd="python plot.py --y_label 'Accuracy(%)' --font_scale 1.75 --y_var_name acc_max --type_plot box --fig_size 10 6 --summarized --results_dir results_all/new_plots/dif_max_acc"
# output_file="max_models_performance_vit_top1"
# cmd="${base_cmd} --input_file results_all/acc/summary_val_acc_level1_main.csv --keep_serials ${serials[*]} --x_label 'Datasets' --x_var_name dataset_name --hue_var_name serial --keep_methods ${all_vit[*]} --keep_datasets ${datasets_array[*]} --output_file ${output_file} --title 'Max Accuracies Variations Between ViT Models (Top1)'"
# echo "Running: ${cmd}"
# eval "${cmd}"

# # Difference of maximum accuracy per model ap_w
# base_cmd="python plot.py --y_label 'Weighted Average Precision(wAP, %)' --font_scale 1.75 --y_var_name acc_max --type_plot box --fig_size 10 6 --summarized --results_dir results_all/new_plots/dif_max_acc"
# output_file="max_models_performance_ap_w"
# cmd="${base_cmd} --input_file results_all/acc/summary_ap_w_main.csv --keep_serials ${serials[*]} --x_label 'Datasets' --x_var_name dataset_name --hue_var_name serial --keep_methods ${all_resnet[*]} --keep_datasets ${datasets_array[*]} --output_file ${output_file} --title 'Max Accuracies Variations Between Resnet Models (wAP)'"
# echo "Running: ${cmd}"
# eval "${cmd}"

# base_cmd="python plot.py --y_label 'Weighted Average Precision(wAP, %)' --font_scale 1.75 --y_var_name acc_max --type_plot box --fig_size 10 6 --summarized --results_dir results_all/new_plots/dif_max_acc"
# output_file="max_models_performance_vit_ap_w"
# cmd="${base_cmd} --input_file results_all/acc/summary_ap_w_main.csv --keep_serials ${serials[*]} --x_label 'Datasets' --x_var_name dataset_name --hue_var_name serial --keep_methods ${all_vit[*]} --keep_datasets ${datasets_array[*]} --output_file ${output_file} --title 'Max Accuracies Variations Between ViT Models (wAP)'"
# echo "Running: ${cmd}"
# eval "${cmd}"

# # All models performance top1
# base_cmd="python plot.py --y_label 'Accuracy(%)' --font_scale 1.75 --y_var_name val_acc_level1 --x_var_name method --x_rotation 30 --type_plot box --loc_legend 'lower left' --hue_var_name dataset_name --fig_size 12 8 --results_dir results_all/new_plots/all_model_performance"
# for serial in "${serials[@]}"; do
#     if [[ "$serial" -eq 24 ]]; then
#         model_array=${all_resnet_onlyfz[*]}
#         prefix="Frozen"
#     else
#         model_array=${all_resnet_nofz[*]}
#         prefix="Fine Tuned"
#     fi
#     output_file="resnet_models_performance_serial_${serial}_top1"
#     cmd="${base_cmd} --input_file data/hierarchical_all.csv --keep_serials ${serial} --keep_methods ${model_array[*]} --output_file ${output_file} --title 'Resnet Models Perfromance ${prefix} (Top1)'"
#     echo "Running: ${cmd}"
#     eval "${cmd}"
# done

# for serial in "${serials[@]}"; do
#     if [[ "$serial" -eq 24 ]]; then
#         model_array=${all_vit_onlyfz[*]}
#         prefix="Frozen"
#     else
#         model_array=${all_vit_nofz[*]}
#         prefix="Fine Tuned"
#     fi
#     output_file="vit_models_performance_serial_${serial}_top1"
#     cmd="${base_cmd} --input_file data/hierarchical_all.csv --keep_serials ${serial} --keep_methods ${model_array[*]} --output_file ${output_file} --title 'ViT Models Perfromance ${prefix} (Top1)'"
#     echo "Running: ${cmd}"
#     eval "${cmd}"
# done

# # All models performance ap_w
# base_cmd="python plot.py --y_label 'Weighted Average Precision(wAP, %)' --font_scale 1.75 --y_var_name ap_w --x_var_name method --x_rotation 30 --type_plot box --loc_legend 'lower left' --hue_var_name dataset_name --fig_size 12 8 --results_dir results_all/new_plots/all_model_performance"
# for serial in "${serials[@]}"; do
#     if [[ "$serial" -eq 24 ]]; then
#         model_array=${all_resnet_onlyfz[*]}
#         prefix="Frozen"
#     else
#         model_array=${all_resnet_nofz[*]}
#         prefix="Fine Tuned"
#     fi
#     output_file="resnet_models_performance_serial_${serial}_ap_w"
#     cmd="${base_cmd} --input_file data/hierarchical_all.csv --keep_serials ${serial} --keep_methods ${model_array[*]} --output_file ${output_file} --title 'Resnet Models Perfromance ${prefix} (wAP)'"
#     echo "Running: ${cmd}"
#     eval "${cmd}"
# done

# for serial in "${serials[@]}"; do
#     if [[ "$serial" -eq 24 ]]; then
#         model_array=${all_vit_onlyfz[*]}
#         prefix="Frozen"
#     else
#         model_array=${all_vit_nofz[*]}
#         prefix="Fine Tuned"
#     fi
#     output_file="vit_models_performance_serial_${serial}_ap_w"
#     cmd="${base_cmd} --input_file data/hierarchical_all.csv --keep_serials ${serial} --keep_methods ${model_array[*]} --output_file ${output_file} --title 'ViT Models Perfromance ${prefix} (wAP)'"
#     echo "Running: ${cmd}"
#     eval "${cmd}"
# done