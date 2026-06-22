# cotton
# soyageing_r1
# soyageing_r1, ratio: 90
python -u tools/train.py --serial 84 --cfg configs/datasets/soyageing_r1/soyageing_r1_pseudo_deit_90.yaml --use_hierarchy --ignore_pl_eval --lr 0.0005 --model_name hideit_base_patch16_224.fb_in1k --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448 --n_cluster_ratio 90
python -u tools/train.py --serial 84 --cfg configs/datasets/soyageing_r1/soyageing_r1_pseudo_vit_b_90.yaml --use_hierarchy --ignore_pl_eval --lr 0.1 --model_name hivit_base_patch16_224.orig_in21k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448 --n_cluster_ratio 90

# soyageing_r3
# soyageing_r3, ratio: 90
python -u tools/train.py --serial 84 --cfg configs/datasets/soyageing_r3/soyageing_r3_pseudo_deit_90.yaml --use_hierarchy --ignore_pl_eval --lr 5e-06 --model_name hideit_base_patch16_224.fb_in1k --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448 --n_cluster_ratio 90
python -u tools/train.py --serial 84 --cfg configs/datasets/soyageing_r3/soyageing_r3_pseudo_vit_b_90.yaml --use_hierarchy --ignore_pl_eval --lr 0.0005 --model_name hivit_base_patch16_224.orig_in21k --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448 --n_cluster_ratio 90

# soyageing_r4
# soyageing_r4, ratio: 90
python -u tools/train.py --serial 84 --cfg configs/datasets/soyageing_r4/soyageing_r4_pseudo_deit_90.yaml --use_hierarchy --ignore_pl_eval --lr 0.1 --model_name hideit_base_patch16_224.fb_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448 --n_cluster_ratio 90
python -u tools/train.py --serial 84 --cfg configs/datasets/soyageing_r4/soyageing_r4_pseudo_vit_b_90.yaml --use_hierarchy --ignore_pl_eval --lr 0.03 --model_name hivit_base_patch16_224.orig_in21k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448 --n_cluster_ratio 90

# soyageing_r5
# soyageing_r5, ratio: 90

# soyageing_r6
# soyglobal
