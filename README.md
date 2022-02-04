# Fast-BAT


This is an official implementation of the paper "Revisiting and Advancing Fast Adversarial Training through the Lens of Bi-level Optimization". Fast-BAT is a new state-of-the-art method for accelerated adversarial training.

## What's in this repository?

This repository serves as the second major contribution to the community besides our proposed Fast-BAT. 
This repository contains various methods for adversarial training as well as attacks for robustness evaluation. 
In addition to neat implementations, this codebase is designed to offer as much flexibility to users as possible, 
including but not limited to abundant datasets, 
model types, 
activation functions, 
loss functions, 
training schedulers, 
optimizers for model parameters, 
(adversarial & standard) training recipes, 
detailed adversarial training settings, and attack methods.

This repository is friendly to researchers and students of all levels. For beginners to adversarial ML, 
the code is easy to understand; for advanced researchers, this code framework is highly-extendable for 
robustness-related research. We hope this codebase sweeps the obstacles on implementation for you and eases your research and study. 


[comment]: <> (> Zhang, Y., Zhang, G., Khanduri, P., Hong, M., Chang, S., & Liu, S. &#40;2021&#41;. Revisiting and Advancing Fast Adversarial Training Through The Lens of Bi-Level Optimization. arXiv preprint arXiv:2112.12376.)

[comment]: <> (This repository includes:)

[comment]: <> (* Training and evaluation code.)

[comment]: <> (* Defense experiments used in the paper.)

[comment]: <> (* Code for baselines used in the paper.)

[comment]: <> (* Code for other frequently adversarial training methods.)


## Requirements
* Install the required python packages:
```
$ python -m pip install -r requirements.py
```
* For dataset ImageNet, please follow the step-by-step instructions in `ImageNet-Download.md` for data downloading and preprocessing.
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

* For training with PGD-2, run command:
```
$ python train.py --mode pgd --dataset <DatasetName> --attack_eps <AttackEps>
```

* For training with PGD-10, run command:
```
$ python train.py --mode pgd --dataset <DatasetName> --attack_eps <AttackEps> --attack_step 10 --epochs 200 --lr_scheduler multistep --lr_max 0.1
```

* For training standard model (PGD-0), run command:
```
$ python train.py --mode pgd --dataset <DatasetName> --attack_step 0 --epochs 200 --lr_scheduler multistep --lr_max 0.1
```
### Parameter choices
* The parameter choices are as following:
    * `<DatasetName>` : `CIFAR10` | `CIFAR100` | `SVHN` | `GTSRB` | `TINY_IMAGENET` | `IMAGENET`
    * `<AttackEps>` : `2~16`(Recommended)

    * For Fast-AT-GA, there is a special parameter `GA_Coefficient`, it can be chosen according to the following table, which is copied from its [official repo](https://github.com/tml-epfl/understanding-fast-adv-training/blob/master/sh/exps_diff_eps_cifar10.sh#L40):

| AttackEps 	| 2     	| 4     	| 6     	| 8     	| 10    	| 12    	| 14    	| 16    	|
|-----------	|-------	|-------	|-------	|-------	|-------	|-------	|-------	|-------	|
| __Ga_Coef__   | 0.036 	| 0.063 	| 0.112 	| 0.200 	| 0.356 	| 0.632 	| 1.124 	| 2.000 	|



* Other possible options:
```
--time_stamp    the flag for each training trial, you can use the current time or whatever you want to specify each training trail. This parameter will be applied to the name of your checkpoints as well as the training report. 
--data_dir,     the path you (want to) store your dataset and be sure to set it to the right folder before training on TinyImageNet (see commands in 'script' folder).
--model_prefix, the path to store your model checkpoints.
--csv_prefix,   the path to store your training result report.
--random_seed,  random seed for pytorch.
--batch_size,   the batch size for your training.
--model_type,   possible choices: 'ResNet', 'PreActResNet', 'WideResNet'
--depth,        possible choices: ResNet(18, 34, 50), PreActResNet(18, 34, 50), WideResNet(16, 28, 34, 70)
```

Your model checkpoints will be saved to folder `./results/<model_prefix>/`, and your training report will be stored at `./results/<csv_prefix>`. By default, they will 
be saved to `./results/checkpoints/` and `./results/accuracy/`. The checkpoint of model in the _last_ epoch and the checkpoint of the _best_ robust accuracy will be stored, corresponding to the without/with early stopping setting. 

## Evaluation
The evaluation provides two classes of attacks: adaptive attack and transfer attack. For adaptive attack, the perturbation generated from
the victim model are tested on itself. For transfer attack, the attack examples are generated based on the surrogate model and tested on 
victim models instead. 
The important parameters for evaluating models (using `evaluation.py`) are listed below:
```
--dataset           [CIFAR10, CIFAR100, SVHN, TINY_IMAGENET] 
--model_path        the path of the checkpoint saved during training.
--model_type        [PreActResNet, ResNet, WideResNet], default to PreActResNet
--depth             [ResNet(18, 34, 50), PreActResNet(18, 34, 50), WideResNet(16, 28, 34, 70)], default to 18
--attack_method     [PGD, AutoAttack], default to PGD
--attack_step       the steps for PGD attack, default to 50
--attack_rs         the restart number for PGD attack, default to 10.
--eps               the attack budgets for evaluation, split by space, e.g. "8 12 16" 
```

When applying __transfer attack__, there are a few more parameters you should pay attention to:
```
--transfer              this parameter identifies the mode of transfer attack and the following parameters are activated. The following three parameters for surrogate models are just like that for victim models above.
--surrogate_model_path  see above
--surrogate_model_type  see above
--surrogate_model_depth see above
```


## Reference

If this code base helps you, please consider citing our paper:

```
@article{zhang2021revisiting,
  title={Revisiting and Advancing Fast Adversarial Training Through The Lens of Bi-Level Optimization},
  author={Zhang, Yihua and Zhang, Guanhuan and Khanduri, Prashant and Hong, Mingyi and Chang, Shiyu and Liu, Sijia},
  journal={arXiv preprint arXiv:2112.12376},
  year={2021}
}
```