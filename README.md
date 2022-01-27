# Fast-BAT


This is an official implementation of paper "Revisiting and Advancing Fast Adversarial Training through the Lens of Bi-level Optimization".
Fast-BAT is a new method for training adversarially robust models with efficiency. 

This repository includes:
* Training and evaluation code.
* Defense experiments used in the paper.
* Code for baselines used in the paper.

## Requirements
* Install the required python packages:
```
$ python -m pip install -r requirements.py
```
* For dataset Tiny-ImageNet, please download and preprocess the dataset with the given command in __sh__ folder:
```
$ bash download_tiny_imagenet.sh
$ bash process_iamge_net.sh
```

## Training
* For training with Fast-BAT, run command:
```
$ python train.py --mode fast_bat --dataset <DatasetName> --attack_eps <AttackEps>  
```

* For training with Fast-AT, run command:
```
$ python train.py --mode fast_at --dataset <DatasetName> --attack_eps <AttackEps>  
```

* For training with Fast-AT-GA, run command:
```
$ python train.py --mode fast_at_ga --dataset <DatasetName> --attack_eps <AttackEps> --ga_coef <GA_Coefficient>
```
* The parameter choices are as following:
    * `<DatasetName>` : `CIFAR10` | `CIFAR100` | `SVHN` | `TINY_IMAGENET`.
    * `<AttackEps>` : `2~16`

    * For Fast-AT-GA, there is a special parameter `GA_Coefficient`, it can be chosen according to the following table, which is copied from its [official repo](https://github.com/tml-epfl/understanding-fast-adv-training/blob/master/sh/exps_diff_eps_cifar10.sh#L40):

| AttackEps 	| 2     	| 4     	| 6     	| 8     	| 10    	| 12    	| 14    	| 16    	|
|-----------	|-------	|-------	|-------	|-------	|-------	|-------	|-------	|-------	|
| __Ga_Coef__   | 0.036 	| 0.063 	| 0.112 	| 0.200 	| 0.356 	| 0.632 	| 1.124 	| 2.000 	|

* Other possible options:
```
--time_stamp    the flag for each training trial, you can use the current time or whatever you want to specify each training trail. This parameter will be used to the name of your checkpoints as well as the training report. 
--data_dir,     the path you (want to) store your dataset and be sure to set to the right folder before training on TinyImageNet.
--model_prefix, the path to store your model checkpoints.
--csv_prefix,   the path to store your training result report.
--random_seed,  random seed for pytorch.
--batch_size,   the batch size for your training.
--model_type,   possible choices: 'ResNet', 'PreActResNet', 'WideResNet'
--depth,        possible choices: ResNet(18, 34, 50), PreActResNet(18, 34, 50), WideResNet(16, 28, 34, 70)
```

Your model checkpoints will be saved to folder `./results/<model_prefix>/`, and your training report will be stored at `./results/<csv_prefix>`. By default, they will 
be saved to `./results/checkpoints/` and `./results/accuracy/`. The checkpoint of model in the last epoch and the checkpoint of the best robust accuracy will be stored, corresponding to the without/with early stopping setting. 

## Evaluation
The evaluation file provides two classes of attacks: adaptive attack and transfer attack. For adaptive attack, the perturbation generated from
the victim model are tested on itself. For transfer attack, the attack examples are generated based on the surrogate model and tested on 
victim models instead. 
The important parameters for evaluate models (using `evaluation.py`) are listed below:
```
--dataset           [CIFAR10, CIFAR100, SVHN, TINY_IMAGENET] 
--model_path        the path of the checkpoint saved during training.
--model_type        [PreActResNet, ResNet, WideResNet], default as PreActResNet
--depth             [ResNet(18, 34, 50), PreActResNet(18, 34, 50), WideResNet(16, 28, 34, 70)], default as 18
--attack_method     [PGD, AutoAttack], default as PGD
--attack_step       the steps for PGD attack, default as 50
--attack_rs         the restart number for PGD attack, default as 10.
--eps               the attack budgets for evaluation, split by space, e.g. "8 12 16" 
```

When applying transfer attack, there are a few more parameters you should pay attention to:
```
--transfer              this parameter identifies the mode of transfer attack and the following parameters are activated. The following three parameters for surrogate models are just like that for victim models.
--surrogate_model_path  see above
--surrogate_model_type  see above
--surrogate_model_depth see above
```
