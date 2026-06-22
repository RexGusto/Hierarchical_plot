# cotton
# cotton
python -u tools/train.py --serial 85 --cfg configs/datasets/cotton.yaml --lr 0.0005 --model_name hideit_base_patch16_224.fb_in1k --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 85 --cfg configs/datasets/cotton.yaml --lr 0.01 --model_name hivit_base_patch16_224.orig_in21k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448

# soyageing_r1
# soyageing_r1
python -u tools/train.py --serial 85 --cfg configs/datasets/soyageing_r1.yaml --lr 0.03 --model_name hideit_base_patch16_224.fb_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 85 --cfg configs/datasets/soyageing_r1.yaml --lr 0.1 --model_name hivit_base_patch16_224.orig_in21k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448

# soyageing_r3
# soyageing_r3
python -u tools/train.py --serial 85 --cfg configs/datasets/soyageing_r3.yaml --lr 5e-05 --model_name hideit_base_patch16_224.fb_in1k --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 85 --cfg configs/datasets/soyageing_r3.yaml --lr 0.0001 --model_name hivit_base_patch16_224.orig_in21k --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448

# soyageing_r4
# soyageing_r4
python -u tools/train.py --serial 85 --cfg configs/datasets/soyageing_r4.yaml --lr 5e-06 --model_name hideit_base_patch16_224.fb_in1k --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 85 --cfg configs/datasets/soyageing_r4.yaml --lr 0.03 --model_name hivit_base_patch16_224.orig_in21k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448

# soyageing_r5
# soyageing_r5
python -u tools/train.py --serial 85 --cfg configs/datasets/soyageing_r5.yaml --lr 0.01 --model_name hideit_base_patch16_224.fb_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 85 --cfg configs/datasets/soyageing_r5.yaml --lr 5e-06 --model_name hivit_base_patch16_224.orig_in21k --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448

# soyageing_r6
# soyageing_r6
python -u tools/train.py --serial 85 --cfg configs/datasets/soyageing_r6.yaml --lr 1e-05 --model_name hideit_base_patch16_224.fb_in1k --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 85 --cfg configs/datasets/soyageing_r6.yaml --lr 5e-05 --model_name hivit_base_patch16_224.orig_in21k --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448

# soygene
# soygene
python -u tools/train.py --serial 85 --cfg configs/datasets/soygene.yaml --lr 0.0001 --model_name hideit_base_patch16_224.fb_in1k --opt adamw --weight_decay 0.05 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 85 --cfg configs/datasets/soygene.yaml --lr 0.03 --model_name hivit_base_patch16_224.orig_in21k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448

# soyglobal
# soyglobal
python -u tools/train.py --serial 85 --cfg configs/datasets/soyglobal.yaml --lr 0.03 --model_name hideit_base_patch16_224.fb_in1k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448
python -u tools/train.py --serial 85 --cfg configs/datasets/soyglobal.yaml --lr 0.03 --model_name hivit_base_patch16_224.orig_in21k --opt sgd --weight_decay 0.0 --cpu_workers 20 --epochs 200 --resize_size 550 --image_size 448

