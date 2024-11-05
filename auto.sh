python3 main.py --model_name 'EfficientNetV2SAutoencoder' --img_size 224 --epochs 100 --alpha 1 --beta 1 --gamma 1 --delta 1 --prefix sh_bn_all
python3 main.py --model_name 'EfficientNetV2SAutoencoder' --img_size 224 --epochs 100 --alpha 0 --beta 0 --gamma 0 --delta 1 --prefix sh_bn_l1
python3 main.py --model_name 'EfficientNetV2SAutoencoder' --img_size 224 --epochs 100 --alpha 0 --beta 0 --gamma 1 --delta 0 --prefix sh_bn_l2
python3 main.py --model_name 'EfficientNetV2SAutoencoder' --img_size 224 --epochs 100 --alpha 1 --beta 0 --gamma 0 --delta 0 --prefix sh_bn_gms
