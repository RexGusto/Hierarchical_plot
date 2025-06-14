#!/bin/bash

# download
# python download_save_wandb_data.py --serials 23 24 --output_file hierarchical_stage1.csv
# python download_save_wandb_data_feature_metrics.py

# merge data from acc, corr metrics and pretraining stats
# python merge_acc_metrics_stats.py

# lr script
# python lr_script.py


# # # accuracy
# python summarize_acc.py

# # cost metrics including flops, no params, trainable params
# python summarize_cost.py
# python summarize_cost.py --acc_to_use val_acc_level2 --output_file cost_val_acc_level2
# python summarize_cost.py --acc_to_use ap_w --output_file cost_ap_w

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

input_csv="data/hierarchical_test.csv"

# name="acc_max_vs_in1kacc"
# base_cmd="python plot.py --y_var_name acc_max --type_plot reg --fig_size 6 4 --summarized --loc_legend 'lower left' --results_dir results_all/new_plots/${name}"
# for dataset in "${datasets_array[@]}";do
#     for serial in "${serials[@]}"; do
#         if [[ "$serial" -eq 24 ]]; then
#             model_array=${all_vit_onlyfz[*]}
#             prefix="FZ"
#         else
#             model_array=${all_vit_nofz[*]}
#             prefix="FT"
#         fi
#         output_file="${name}_reg_resnet_${dataset}_${prefix}"
#         cmd="${base_cmd} --input_file results_all/acc/summarized_acc_hierarchical_main.csv --title '' --x_var_name acc_in1k --keep_datasets ${dataset} --keep_serials ${serial} --keep_methods ${all_resnet[*]} --output_file ${output_file}"
#         echo "Running: ${cmd}"
#         eval "${cmd}"
#     done
# done

# for dataset in "${datasets_array[@]}";do
#     for serial in "${serials[@]}"; do
#         if [[ "$serial" -eq 24 ]]; then
#             model_array=${all_vit_onlyfz[*]}
#             prefix="FZ"
#         else
#             model_array=${all_vit_nofz[*]}
#             prefix="FT"
#         fi
#         output_file="${name}_reg_vit_${dataset}_${prefix}"
#         cmd="${base_cmd} --input_file results_all/acc/summarized_acc_hierarchical_main.csv --title '' --x_var_name acc_in1k --keep_datasets ${dataset} --keep_serials ${serial} --keep_methods ${all_vit[*]} --output_file ${output_file}"
#         echo "Running: ${cmd}"
#         eval "${cmd}"
#     done
# done
# # Per architecture (combined)
# for dataset in "${datasets_array[@]}";do
#     for serial in "${serials[@]}"; do
#         if [[ "$serial" -eq 24 ]]; then
#             model_used=${all_models_fz[*]}
#             add_cmd="--y_var_name acc_max --x_var_name acc_in1k"
#             prefix="FZ"
#         else
#             model_used=${all_models_ft[*]}
#             add_cmd="--y_var_name acc_max --x_var_name acc_in1k"
#             prefix="FT"
#         fi
#         output_file="${name}_reg_both_${dataset}_${prefix}"
#         cmd="${base_cmd} --input_file results_all/acc/summarized_acc_hierarchical_main.csv ${add_cmd} --title '' --keep_datasets ${dataset} --keep_serials ${serial} --keep_methods ${model_used[*]} --output_file ${output_file}"
#         echo "Running: ${cmd}"
#         eval "${cmd}"
#     done
# done

# #ft & fz combined
# for dataset in "${datasets_array[@]}";do
#     output_file="${name}_reg_both_${dataset}_FT_FZ"
#     add_cmd="--y_var_name acc_max --x_var_name acc_in1k"
#     cmd="${base_cmd} --input_file results_all/acc/summarized_acc_hierarchical_main.csv ${add_cmd} --title '' --keep_datasets ${dataset} --keep_serials ${serials[*]} --keep_methods ${all_models[*]} --output_file ${output_file}"
#     echo "Running: ${cmd}"
#     eval "${cmd}"
# done

# # All models performance
# base_cmd="python plot.py --y_label 'Accuracy(%)' --y_var_name acc --x_var_name method --x_rotation 45 --type_plot box --loc_legend 'lower left' --hue_var_name dataset_name --fig_size 12 8 --results_dir results_all/new_plots/all_model_performance"
# for serial in "${serials[@]}"; do
#     if [[ "$serial" -eq 24 ]]; then
#         model_array=${all_resnet_onlyfz[*]}
#         prefix="Frozen"
#     else
#         model_array=${all_resnet_nofz[*]}
#         prefix="Fine Tuned"
#     fi
#     output_file="resnet_models_performance_serial_${serial}"
#     cmd="${base_cmd} --input_file ${input_csv} --keep_serials ${serial} --keep_methods ${model_array[*]} --output_file ${output_file} --title 'Resnet Models Perfromance ${prefix}'"
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
#     output_file="vit_models_performance_serial_${serial}"
#     cmd="${base_cmd} --input_file ${input_csv} --keep_serials ${serial} --keep_methods ${model_array[*]} --output_file ${output_file} --title 'ViT Models Perfromance ${prefix}'"
#     echo "Running: ${cmd}"
#     eval "${cmd}"
# done

# top3_best_models=('hiresnet50.in1k_mocov3' 'hiresnet50.in1k_swav' 'hiresnet50.tv2_in1k' 'hiresnet50.in1k_spark' 'hiresnet50.in21k_miil' 'hivit_base_patch16_224.orig_in21k' 'hideit3_base_patch16_224.fb_in22k_ft_in1k' 'hivit_base_patch16_siglip_224.v2_webli' 'hivit_base_patch16_224.dino' 'hivit_base_patch16_clip_224.laion2b')
# # ViT best vs resnet best
# base_cmd="python plot.py --y_label 'Accuracy(%)' --y_var_name acc --hue_var_name dataset_name --x_rotation 15 --x_var_name method --type_plot box --loc_legend 'lower left' --hue_var_name dataset_name --fig_size 12 8 --results_dir results_all/new_plots/all_model_performance"
# output_file="top3_best_models_plot"
# cmd="${base_cmd} --input_file ${input_csv} --keep_serials 23 --keep_methods ${top3_best_models[*]} --output_file ${output_file} --title 'Top-3 Best & Top-2 Worst for Both Architecture' --keep_datasets ${datasets_array[*]}"
# echo "Running: ${cmd}"
# eval "${cmd}"

# Difference of maximum accuracy per model 
# base_cmd="python plot.py --y_label 'Accuracy(%)' --y_var_name acc_max --type_plot box --fig_size 10 6 --summarized --results_dir results_all/new_plots/dif_max_acc"
# output_file="max_models_performance"
# cmd="${base_cmd} --input_file results_all/acc/summarized_acc_hierarchical_main.csv --keep_serials ${serials[*]} --x_label 'Datasets' --x_var_name dataset_name --hue_var_name serial --keep_methods ${all_resnet[*]} --keep_datasets ${datasets_array[*]} --output_file ${output_file} --title 'Max Accuracies Difference Between Resnet Models'"
# echo "Running: ${cmd}"
# eval "${cmd}"

# base_cmd="python plot.py --y_label 'Accuracy(%)' --y_var_name acc_max --type_plot box --fig_size 10 6 --summarized --results_dir results_all/new_plots/dif_max_acc"
# output_file="max_models_performance_vit"
# cmd="${base_cmd} --input_file results_all/acc/summarized_acc_hierarchical_main.csv --keep_serials ${serials[*]} --x_label 'Datasets' --x_var_name dataset_name --hue_var_name serial --keep_methods ${all_vit[*]} --keep_datasets ${datasets_array[*]} --output_file ${output_file} --title 'Max Accuracies Difference Between ViT Models'"
# echo "Running: ${cmd}"
# eval "${cmd}"

accuracies=('acc' 'acc_2')
methods=('resnet' 'vit')
metrics=('MSC_train' 'MSC_test' 'V_intra_train' 'V_intra_test' 'S_inter_train' 'S_inter_test'
         'cis_clustering_train' 'cis_clustering_test' 'cis_spectral_train' 'cis_spectral_test''cka_avg_train' 
         'cka_avg_test' 'dist_avg_train' 'dist_avg_test' 'dist_norm_avg_train' 'dist_norm_avg_test' 'l2_norm_avg_train' 'l2_norm_avg_test'
         'cka_0_train' 'cka_0_test' 'cka_high_mean_train' 'cka_mid_mean_train' 'cka_low_mean_train'
         'cka_high_mean_test' 'cka_mid_mean_test' 'cka_low_mean_test' 'clustering_diversity_train' 'spectral_diversity_train')
         
for accuracy in "${accuracies[@]}"; do
    if [[ "$accuracy" == "acc_2" ]]; then
        csv_suffix="_lvl2"
    else
        csv_suffix=""
    fi

    for metric in "${metrics[@]}"; do
        name="${accuracy}_max_vs_${metric,,}"
        base_cmd="python plot.py --type_plot reg --fig_size 6 4 --summarized --results_dir results_all/new_plots/${name}"

        # Plots for ViT+FT, ViT+FZ, RN+FT, RN+FZ
        for dataset in "${datasets_array[@]}";do
            for serial in "${serials[@]}"; do
                for method in "${methods[@]}"; do
                    if [[ "$serial" -eq 24 ]]; then
                        prefix="FZ"
                        # add_cmd="--y_var_name last_acc --x_var_name ${metric}"
                        add_cmd="--y_var_name ${accuracy}_max --x_var_name ${metric}"
                    else
                        prefix="FT"
                        add_cmd="--y_var_name ${accuracy}_max --x_var_name ${metric}_ft"
                    fi

                    if [[ "$method" == "resnet" ]]; then
                        model_used=${all_resnet[*]}
                        model_label="RN"
                    else
                        model_used=${all_vit[*]}
                        model_label="ViT"
                    fi

                    output_file="${name}_${method}_${dataset}_${prefix}"
                    cmd="${base_cmd} --input_file results_all/acc/summarized_acc_hierarchical_main${csv_suffix}.csv ${add_cmd} --title '' --keep_datasets ${dataset} --keep_serials ${serial} --keep_methods ${model_used[*]} --output_file ${output_file}"
                    echo "Running: ${cmd}"
                    eval "${cmd}"
                done
            done
        done

        # Plots for ViT+RN+FT, ViT+RN+FZ
        # acc_ft vs metrics_ft & acc_fz vs metric_fz
        for dataset in "${datasets_array[@]}";do
            for serial in "${serials[@]}"; do
                # acc_fz vs metric_fz
                if [[ "$serial" -eq 24 ]]; then
                    model_used=${all_models_fz[*]}
                    # add_cmd="--y_var_name last_acc --x_var_name ${metric}"
                    add_cmd="--y_var_name ${accuracy}_max --x_var_name ${metric}"
                    prefix="FZ"
                # acc_ft vs metrics_ft
                else
                    model_used=${all_models_ft[*]}
                    add_cmd="--y_var_name ${accuracy}_max --x_var_name ${metric}_ft"
                    prefix="FT"
                fi
                output_file="${name}_both_${dataset}_${prefix}"
                cmd="${base_cmd} --input_file results_all/acc/summarized_acc_hierarchical_main${csv_suffix}.csv ${add_cmd} --title '' --keep_datasets ${dataset} --keep_serials ${serial} --keep_methods ${model_used[*]} --output_file ${output_file}"
                echo "Running: ${cmd}"
                eval "${cmd}"
            done
        done

        # New plot:
        for dataset in "${datasets_array[@]}";do
            for serial in "${serials[@]}"; do
                # acc_fz vs metric_ft
                if [[ "$serial" -eq 24 ]]; then
                    model_used=${all_models_fz[*]}
                    # add_cmd="--y_var_name last_acc --x_var_name ${metric}"
                    add_cmd="--y_var_name ${accuracy}_max --x_var_name ${metric}_ft"
                    alter_name="${accuracy}_fz_vs_${metric,,}_ft"
                # acc_ft vs metrics_fz
                else
                    model_used=${all_models_ft[*]}
                    add_cmd="--y_var_name ${accuracy}_max --x_var_name ${metric}"
                    alter_name="${accuracy}_ft_vs_${metric,,}_fz"
                fi
                output_file="${alter_name}_both_${dataset}_FZ_v_FT"
                cmd="${base_cmd} --input_file results_all/acc/summarized_acc_hierarchical_main${csv_suffix}.csv ${add_cmd} --title '' --keep_datasets ${dataset} --keep_serials ${serial} --keep_methods ${model_used[*]} --output_file ${output_file}"
                echo "Running: ${cmd}"
                eval "${cmd}"
            done
        done

        # Plots for ViT+RN+FT+FZ 
        for dataset in "${datasets_array[@]}";do
            output_file="${name}_both_${dataset}_FT_FZ"
            add_cmd="--y_var_name ${accuracy}_matched --x_var_name ${metric}_matched"
            cmd="${base_cmd} --input_file results_all/acc/summarized_acc_hierarchical_main${csv_suffix}.csv ${add_cmd} --title '' --keep_datasets ${dataset} --keep_serials ${serials[*]} --keep_methods ${all_models[*]} --output_file ${output_file}"
            echo "Running: ${cmd}"
            eval "${cmd}"
        done
    done
done

# Case for very last layer of CKA.
# Resnet uses 16 layers, while ViT uses 12 layers
states=('train' 'test')
for accuracy in "${accuracies[@]}"; do
    if [[ "$accuracy" == "acc_2" ]]; then
        csv_suffix="_lvl2"
    fi
    for state in "${states[@]}"; do
        name="acc_max_vs_cka_last_layer_${state}"
        base_cmd="python plot.py --type_plot reg --fig_size 6 4 --summarized --results_dir results_all/new_plots/${name}"

        # Plots for ViT+FT, ViT+FZ, RN+FT, RN+FZ
        for dataset in "${datasets_array[@]}"; do
            for serial in "${serials[@]}"; do
                for method in "${methods[@]}"; do
                    if [[ "$method" == "resnet" ]]; then
                        model_used=${all_resnet[*]}
                        metric=cka_15_${state}
                    else
                        model_used=${all_vit[*]}
                        metric=cka_11_${state}
                    fi

                    if [[ "$serial" -eq 24 ]]; then
                        prefix="FZ"
                        add_cmd="--y_var_name ${accuracy}_max --x_var_name ${metric}"
                    else
                        prefix="FT"
                        # add_cmd="--y_var_name last_acc --x_var_name ${metric}_ft"
                        add_cmd="--y_var_name ${accuracy}_max --x_var_name ${metric}_ft"
                    fi

                    output_file="${name}_reg_${method}_${dataset}_${prefix}"
                    cmd="${base_cmd} --input_file results_all/acc/summarized_acc_hierarchical_main${csv_suffix}.csv ${add_cmd} --title '' --keep_serials ${serial} --keep_datasets ${dataset} --keep_methods ${model_used[*]} --output_file ${output_file}"
                    echo "Running: ${cmd}"
                    eval "${cmd}"
                done
            done
        done

        # Plots for ViT+RN+FT, ViT+RN+FZ
        # acc_ft vs metrics_ft & acc_fz vs metric_fz
        for dataset in "${datasets_array[@]}";do
            for serial in "${serials[@]}"; do
                if [[ "$serial" -eq 24 ]]; then
                    model_used=${all_models_fz[*]}
                    # add_cmd="--y_var_name last_acc --x_var_name cka_last_matched_${state}"
                    add_cmd="--y_var_name ${accuracy}_max --x_var_name cka_last_matched_${state}"
                    prefix="FZ"
                else
                    model_used=${all_models_ft[*]}
                    add_cmd="--y_var_name ${accuracy}_max --x_var_name cka_last_matched_${state}_ft"
                    prefix="FT"
                fi
                output_file="${name}_reg_both_${dataset}_${prefix}"
                cmd="${base_cmd} --input_file results_all/acc/summarized_acc_hierarchical_main${csv_suffix}.csv ${add_cmd} --title '' --keep_datasets ${dataset} --keep_serials ${serial} --keep_methods ${model_used[*]} --output_file ${output_file}"
                echo "Running: ${cmd}"
                eval "${cmd}"
            done
        done

        # New plot:
        for dataset in "${datasets_array[@]}";do
            for serial in "${serials[@]}"; do
                # acc_fz vs metric_ft
                if [[ "$serial" -eq 24 ]]; then
                    model_used=${all_models_fz[*]}
                    # add_cmd="--y_var_name last_acc --x_var_name cka_last_matched_${state}"
                    add_cmd="--y_var_name ${accuracy}_max --x_var_name cka_last_matched_${state}_ft"
                    alter_name="${accuracy}_max_fz_vs_cka_last_layer_${state}_ft"
                # acc_ft vs metric_fz
                else
                    model_used=${all_models_ft[*]}
                    add_cmd="--y_var_name ${accuracy}_max --x_var_name cka_last_matched_${state}"
                    alter_name="${accuracy}_max_ft_vs_cka_last_layer_${state}_fz"
                fi
                output_file="${alter_name}_reg_both_${dataset}_FZ_v_FT"
                cmd="${base_cmd} --input_file results_all/acc/summarized_acc_hierarchical_main${csv_suffix}.csv ${add_cmd} --title '' --keep_datasets ${dataset} --keep_serials ${serial} --keep_methods ${model_used[*]} --output_file ${output_file}"
                echo "Running: ${cmd}"
                eval "${cmd}"
            done
        done

        # Plots for ViT+RN+FT+FZ 
        for dataset in "${datasets_array[@]}";do
            output_file="${name}_reg_both_${dataset}_FT_FZ"
            add_cmd="--y_var_name acc_matched --x_var_name cka_last_matched_${state}_matched"
            cmd="${base_cmd} --input_file results_all/acc/summarized_acc_hierarchical_main.csv ${add_cmd} --title '' --keep_datasets ${dataset} --keep_serials ${serials[*]} --keep_methods ${all_models[*]} --output_file ${output_file}"
            echo "Running: ${cmd}"
            eval "${cmd}"
        done
    done
done
