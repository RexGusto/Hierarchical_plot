# aircraft
# aircraft
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.01 --model_name hideit3_base_patch16_224.fb_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.01 --model_name hideit3_base_patch16_224.fb_in22k_ft_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.01 --model_name hideit_base_patch16_224.fb_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.0005 --model_name hiresnet50.a1_in1k --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.03 --model_name hiresnet50.fb_ssl_yfcc100m_ft_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.03 --model_name hiresnet50.fb_swsl_ig1b_ft_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.03 --model_name hiresnet50.gluon_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.0001 --model_name hiresnet50.in1k_mocov3 --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.0001 --model_name hiresnet50.in1k_spark --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.03 --model_name hiresnet50.in1k_supcon --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 5e-05 --model_name hiresnet50.in1k_swav --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.0001 --model_name hiresnet50.in21k_miil --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.03 --model_name hiresnet50.tv2_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.01 --model_name hiresnet50.tv_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.003 --model_name hivit_base_patch16_224.dino --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.01 --model_name hivit_base_patch16_224.in1k_mocov3 --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.01 --model_name hivit_base_patch16_224.mae --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.01 --model_name hivit_base_patch16_224.orig_in21k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.01 --model_name hivit_base_patch16_224_miil.in21k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 1e-05 --model_name hivit_base_patch16_clip_224.laion2b --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.003 --model_name hivit_base_patch16_siglip_224.v2_webli --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.0005 --model_name resnet50.a1_in1k --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.01 --model_name resnet50.fb_ssl_yfcc100m_ft_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.01 --model_name resnet50.fb_swsl_ig1b_ft_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.03 --model_name resnet50.gluon_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.03 --model_name resnet50.tv2_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.03 --model_name resnet50.tv_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.01 --model_name vit_base_patch16_224.orig_in21k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/aircraft.yaml --lr 0.01 --model_name vit_base_patch16_224_miil.in21k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448

# cars
# cars
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.01 --model_name hideit3_base_patch16_224.fb_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.03 --model_name hideit3_base_patch16_224.fb_in22k_ft_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.03 --model_name hideit_base_patch16_224.fb_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.0001 --model_name hiresnet50.a1_in1k --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.03 --model_name hiresnet50.fb_ssl_yfcc100m_ft_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.03 --model_name hiresnet50.fb_swsl_ig1b_ft_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.03 --model_name hiresnet50.gluon_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 5e-05 --model_name hiresnet50.in1k_mocov3 --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.01 --model_name hiresnet50.in1k_spark --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.03 --model_name hiresnet50.in1k_supcon --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.0001 --model_name hiresnet50.in1k_swav --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.0001 --model_name hiresnet50.in21k_miil --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.03 --model_name hiresnet50.tv2_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.01 --model_name hiresnet50.tv_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.003 --model_name hivit_base_patch16_224.dino --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.01 --model_name hivit_base_patch16_224.in1k_mocov3 --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.01 --model_name hivit_base_patch16_224.mae --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.03 --model_name hivit_base_patch16_224.orig_in21k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.03 --model_name hivit_base_patch16_224_miil.in21k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 5e-06 --model_name hivit_base_patch16_clip_224.laion2b --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.003 --model_name hivit_base_patch16_siglip_224.v2_webli --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.01 --model_name resnet50.fb_ssl_yfcc100m_ft_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.01 --model_name resnet50.fb_swsl_ig1b_ft_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.03 --model_name resnet50.gluon_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.0001 --model_name resnet50.tv2_in1k --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.01 --model_name resnet50.tv_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.03 --model_name vit_base_patch16_224.orig_in21k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cars.yaml --lr 0.01 --model_name vit_base_patch16_224_miil.in21k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448

# cub
# cub
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.03 --model_name hideit3_base_patch16_224.fb_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.01 --model_name hideit3_base_patch16_224.fb_in22k_ft_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.03 --model_name hideit_base_patch16_224.fb_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.0001 --model_name hiresnet50.a1_in1k --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.03 --model_name hiresnet50.fb_ssl_yfcc100m_ft_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.01 --model_name hiresnet50.fb_swsl_ig1b_ft_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.03 --model_name hiresnet50.gluon_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.03 --model_name hiresnet50.in1k_mocov3 --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.01 --model_name hiresnet50.in1k_spark --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.01 --model_name hiresnet50.in1k_supcon --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.0001 --model_name hiresnet50.in1k_swav --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.003 --model_name hiresnet50.in21k_miil --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.03 --model_name hiresnet50.tv2_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.01 --model_name hiresnet50.tv_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.003 --model_name hivit_base_patch16_224.dino --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.01 --model_name hivit_base_patch16_224.in1k_mocov3 --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.01 --model_name hivit_base_patch16_224.mae --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.03 --model_name hivit_base_patch16_224.orig_in21k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.01 --model_name hivit_base_patch16_224_miil.in21k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 5e-06 --model_name hivit_base_patch16_clip_224.laion2b --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.01 --model_name hivit_base_patch16_siglip_224.v2_webli --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.03 --model_name deit3_base_patch16_224.fb_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.03 --model_name deit_base_patch16_224.fb_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.0001 --model_name resnet50.a1_in1k --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.01 --model_name resnet50.fb_ssl_yfcc100m_ft_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.01 --model_name resnet50.fb_swsl_ig1b_ft_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.01 --model_name resnet50.gluon_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.01 --model_name resnet50.tv2_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.01 --model_name resnet50.tv_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.03 --model_name vit_base_patch16_224.orig_in21k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 59 --cfg configs/datasets/cub.yaml --lr 0.01 --model_name vit_base_patch16_224_miil.in21k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448

