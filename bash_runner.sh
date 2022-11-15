#!/bin/bash

# ----- CLASSIFIER -----

# Cifar 10
python train_classification.py --classification_model_name resnet --dataset cifar10 --num_classes 10 --epochs 50 --batch_size 256 --save_model ;
python train_classification.py --classification_model_name vit --dataset cifar10 --num_classes 10 --epochs 50 --batch_size 256 --save_model ;


# SVHN
python train_classification.py --classification_model_name resnet --dataset svhn --num_classes 10 --epochs 50 --batch_size 256 --save_model ;
python train_classification.py --classification_model_name vit --dataset svhn --num_classes 10 --epochs 50 --batch_size 256 --save_model ;


# Cifar 100
#python train_classification.py --classification_model_name resnet --dataset cifar100 --num_classes 100 --epochs 100 --batch_size 256 --save_model ;
#python train_classification.py --classification_model_name vit --dataset cifar100 --num_classes 100 --epochs 100 --batch_size 256 --save_model ;



# ----- DETECTOR -----

# -- ViT --

# ID = Cifar 10
#python train_ood_detector.py --epochs 50 --iterations 10 --restarts 2 --detector_model_name vit --num_heads 4 --num_layers 5 --data_id cifar10 --data_ood svhn --save_model ;
#python train_ood_detector.py --epochs 50 --iterations 10 --restarts 2 --detector_model_name vit --num_heads 4 --num_layers 5 --data_id cifar10 --data_ood cifar100 --save_model ;

# ID = Cifar 100
#python train_ood_detector.py --epochs 100 --iterations 10 --restarts 2 --detector_model_name vit --num_heads 4 --num_layers 5 --data_id cifar100 --data_ood svhn --save_model ;
#python train_ood_detector.py --epochs 100 --iterations 10 --restarts 2 --detector_model_name vit --num_heads 4 --num_layers 5 --data_id cifar100 --data_ood cifar10 --save_model ;

# ID = SVHN
#python train_ood_detector.py --epochs 50 --iterations 10 --restarts 2 --detector_model_name vit --num_heads 4 --num_layers 5 --data_id svhn --data_ood cifar10 --save_model ;
#python train_ood_detector.py --epochs 50 --iterations 10 --restarts 2 --detector_model_name vit --num_heads 4 --num_layers 5 --data_id svhn --data_ood cifar100 --save_model ;


# -- ResNet --

# ID = Cifar 10
#python train_ood_detector.py --epochs 50 --iterations 10 --restarts 2 --detector_model_name resnet --data_id cifar10 --data_ood svhn --save_model ;
#python train_ood_detector.py --epochs 50 --iterations 10 --restarts 2 --detector_model_name resnet --data_id cifar10 --data_ood cifar100 --save_model ;

# ID = Cifar 100
#python train_ood_detector.py --epochs 100 --iterations 10 --restarts 2 --detector_model_name resnet --data_id cifar100 --data_ood cifar10 --save_model ;
#python train_ood_detector.py --epochs 100 --iterations 10 --restarts 2 --detector_model_name resnet --data_id cifar100 --data_ood svhn --save_model ;

# ID = SVHN
#python train_ood_detector.py --epochs 50 --iterations 10 --restarts 2 --detector_model_name resnet --data_id svhn --data_ood cifar10 --save_model ;
#python train_ood_detector.py --epochs 50 --iterations 10 --restarts 2 --detector_model_name resnet --data_id svhn --data_ood cifar100 --save_model ;



# -- Mininet --

# ID = Cifar 10
#python train_ood_detector.py --epochs 50 --iterations 10 --restarts 2 --detector_model_name mininet --num_heads 4 --num_layers 5 --data_id cifar10 --data_ood svhn --save_model ;
#python train_ood_detector.py --epochs 50 --iterations 10 --restarts 2 --detector_model_name mininet --num_heads 4 --num_layers 5 --data_id cifar10 --data_ood cifar100 --save_model ;

# ID = Cifar 100
#python train_ood_detector.py --epochs 100 --iterations 10 --restarts 2 --detector_model_name mininet --num_heads 4 --num_layers 5 --data_id cifar100 --data_ood svhn --save_model ;
#python train_ood_detector.py --epochs 100 --iterations 10 --restarts 2 --detector_model_name mininet --num_heads 4 --num_layers 5 --data_id cifar100 --data_ood cifar10 --save_model ;

# ID = SVHN
#python train_ood_detector.py --epochs 50 --iterations 10 --restarts 2 --detector_model_name mininet --num_heads 4 --num_layers 5 --data_id svhn --data_ood cifar10 --save_model ;
#python train_ood_detector.py --epochs 50 --iterations 10 --restarts 2 --detector_model_name mininet --num_heads 4 --num_layers 5 --data_id svhn --data_ood cifar100 --save_model ;



wait