import os
from tqdm import tqdm as tqdm
import argparse
import copy

from model_zoo import *
from utils.general_utils import write_csv_rows
from datasets import *
from attack import AutoAttack
from utils.context import ctx_noparamgrad
from attack.pgd_attack_restart import attack_pgd_restart

parser = argparse.ArgumentParser(description='Evaluation for Cifar10 Dataset')
parser.add_argument('--model_path', required=True)
parser.add_argument('--model_normalize', default=True, type=bool)
parser.add_argument("--batch_size", default=200, type=int,
                    help="Batch size used in the training and validation loop.")
parser.add_argument('--device', default="cuda:0")
parser.add_argument('--dataset', default="CIFAR10", choices=["CIFAR10", "CIFAR100"])
parser.add_argument('--model_type', default='PreActResNet', choices=['WideResNet', 'ResNet', 'PreActResNet'])
parser.add_argument('--depth', default=18, type=int, help="Number of layers.")
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate.")
parser.add_argument('--act_fn', default="relu", choices=["relu", "softplus", "swish"],
                    help="choose the activation function for your model")
parser.add_argument('--eps', default=[10, 12, 14, 16], type=int, nargs="+")

parser.add_argument('--surrogate_model_path', default=None)
parser.add_argument('--surrogate_model_type', default='PreActResNet', choices=['WideResNet', 'ResNet', 'PreActResNet'])
parser.add_argument('--surrogate_model_depth', default=50, type=int, help="Number of layers of surrogate model.")
parser.add_argument("--surrogate_model_dropout", default=0.1, type=float, help="Surrogate model dropout rate.")

parser.add_argument('--attack_step', default=50, type=int,
                    help='attack steps for training (default: 50)')
parser.add_argument('--attack_rs', default=10, type=int,
                    help='attack restart number for evaluation')
parser.add_argument('--attack_method', default='PGD', choices=['PGD', 'AutoAttack'])
parser.add_argument('--pgd_no_sign', default=False, action="store_true")
parser.add_argument('--transfer', default=False, action="store_true", help="Do you want to apply transfer attack?")

args = parser.parse_args()


def evaluation(model_path):
    device = args.device
    attack_method = args.attack_method
    attack_step = args.attack_step
    dataset = args.dataset
    attack_rs = args.attack_rs

    print(model_path)
    print(args.eps)

    ########################## dataset and model ##########################
    if args.dataset == "CIFAR10":
        train_dl, val_dl, test_dl, norm_layer = cifar10_dataloader(batch_size=args.batch_size)
        num_classes = 10
        conv1_size = 3
    elif args.dataset == "CIFAR100":
        train_dl, val_dl, test_dl, norm_layer = cifar100_dataloader(batch_size=args.batch_size)
        num_classes = 100
        conv1_size = 3
    else:
        raise NotImplementedError("Invalid Dataset")

    eval_dl = test_dl

    if args.act_fn == "relu":
        activation_fn = nn.ReLU
    elif args.act_fn == "softplus":
        activation_fn = nn.Softplus
    elif args.act_fn == "swish":
        activation_fn = Swish
    else:
        raise NotImplementedError("Unsupported activation function!")

    if args.model_type == "WideResNet":
        if args.depth == 16:
            model = WRN_16_8(num_classes=num_classes, conv1_size=conv1_size, dropout=args.dropout,
                             activation_fn=activation_fn)
        elif args.depth == 28:
            model = WRN_28_10(num_classes=num_classes, conv1_size=conv1_size, dropout=args.dropout,
                              activation_fn=activation_fn)
        elif args.depth == 34:
            model = WRN_34_10(num_classes=num_classes, conv1_size=conv1_size, dropout=args.dropout,
                              activation_fn=activation_fn)
        elif args.depth == 70:
            model = WRN_70_16(num_classes=num_classes, conv1_size=conv1_size, dropout=args.dropout,
                              activation_fn=activation_fn)
        else:
            raise NotImplementedError("Unsupported WideResNet!")
    elif args.model_type == "PreActResNet":
        if args.depth == 18:
            model = PreActResNet18(num_classes=num_classes, conv1_size=conv1_size, activation_fn=activation_fn)
        elif args.depth == 34:
            model = PreActResNet34(num_classes=num_classes, conv1_size=conv1_size, activation_fn=activation_fn)
        else:
            model = PreActResNet50(num_classes=num_classes, conv1_size=conv1_size, activation_fn=activation_fn)
    elif args.model_type == "ResNet":
        if args.depth == 18:
            model = ResNet18(num_classes=num_classes, conv1_size=conv1_size, activation_fn=activation_fn)
        elif args.depth == 34:
            model = ResNet34(num_classes=num_classes, conv1_size=conv1_size, activation_fn=activation_fn)
        else:
            model = ResNet50(num_classes=num_classes, conv1_size=conv1_size, activation_fn=activation_fn)
    else:
        raise NotImplementedError("Unsupported Model Type!")
    model.normalize = norm_layer

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model = model.to(device)
    model_name = ".".join(model_path.split('/')[-1].split('.')[:-1])

    if args.transfer:

        if args.surrogate_model_path is None:
            raise ValueError("You choose transfer attack but forget to provide surrogate model path.")

        if args.surrogate_model_type == "WideResNet":
            if args.surrogate_model_depth == 16:
                surrogate_model = WRN_16_8(dropout=args.surrogate_model_dropout, num_classes=num_classes,
                                           activation_fn=activation_fn, conv1_size=conv1_size)
            elif args.surrogate_model_depth == 28:
                surrogate_model = WRN_28_10(dropout=args.surrogate_model_dropout, num_classes=num_classes,
                                            activation_fn=activation_fn, conv1_size=conv1_size)
            elif args.surrogate_model_depth == 34:
                surrogate_model = WRN_34_10(dropout=args.surrogate_model_dropout, num_classes=num_classes,
                                            activation_fn=activation_fn, conv1_size=conv1_size)
            elif args.surrogate_model_depth == 70:
                surrogate_model = WRN_70_16(dropout=args.surrogate_model_dropout, num_classes=num_classes,
                                            activation_fn=activation_fn, conv1_size=conv1_size)
            else:
                raise NameError("Unsupported WideResNet!")
        elif args.surrogate_model_type == "PreActResNet":
            if args.surrogate_model_depth == 18:
                surrogate_model = PreActResNet18(num_classes=num_classes, conv1_size=conv1_size,
                                                 activation_fn=activation_fn)
            elif args.surrogate_model_depth == 34:
                surrogate_model = PreActResNet34(num_classes=num_classes, conv1_size=conv1_size,
                                                 activation_fn=activation_fn)
            else:
                surrogate_model = PreActResNet50(num_classes=num_classes, conv1_size=conv1_size,
                                                 activation_fn=activation_fn)
        elif args.surrogate_model_type == "ResNet":
            if args.surrogate_model_depth == 18:
                surrogate_model = ResNet18(activation_fn=nn.ReLU(), num_classes=num_classes)
            elif args.surrogate_model_depth == 34:
                surrogate_model = ResNet34(activation_fn=nn.ReLU(), num_classes=num_classes)
            else:
                surrogate_model = ResNet50(activation_fn=nn.ReLU(), num_classes=num_classes)
        else:
            raise NotImplementedError("Unsupported Model Type!")

        surrogate_model.normalize = norm_layer

        surrogate_model.load_state_dict(torch.load(args.surrogate_model_path, map_location=torch.device(device)))
        surrogate_model = surrogate_model.to(device)
        surrogate_model_name = ".".join(args.surrogate_model_path.split('/')[-1].split('.')[:-1])
    else:
        surrogate_model = copy.deepcopy(model)

    epsilon = args.eps

    if not os.path.exists('results/evaluation/'):
        os.mkdir('results/evaluation/')

    if not args.transfer:
        file_path = f'results/evaluation/Evaluation_{dataset}_{attack_method}_attack_{model_name}.csv'
    else:
        file_path = f'results/evaluation/Evaluation_Transfer_{dataset}_{attack_method}_attack_{model_name}_{surrogate_model_name}.csv'

    result_list = []
    csv_row_list = [epsilon]

    model.eval()
    surrogate_model.eval()
    correct = 0
    total = 0
    for ii, (images, labels) in tqdm(enumerate(eval_dl)):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Natural accuracy: %.2f %%' % (100. * (correct / total).cpu().item()))

    for eps in epsilon:
        if attack_method == 'PGD':
            print(f"{attack_method}-{attack_step}-{attack_rs}, eps:{eps}.")
        elif attack_method == 'AutoAttack':
            print(f"{attack_method}, eps:{eps}.")

        eps = eps / 255

        if attack_method == 'PGD':
            attack_total = 0
            attack_correct = 0
            for ii, (data, label) in tqdm(enumerate(eval_dl)):
                data = data.type(torch.FloatTensor)
                data = data.to(device)
                label = label.to(device)
                with ctx_noparamgrad(surrogate_model):
                    perturbed_data = attack_pgd_restart(
                        model=surrogate_model,
                        X=data,
                        y=label,
                        eps=eps,
                        alpha=eps / 4,
                        attack_iters=attack_step,
                        n_restarts=attack_rs,
                        rs=True,
                        verbose=False,
                        linf_proj=True,
                        l2_proj=False,
                        l2_grad_update=False,
                        cuda=True
                    ) + data

                score = model(perturbed_data)
                _, predicted = torch.max(score, 1)
                attack_total += label.cpu().size(0)
                attack_correct += (predicted == label).sum()
        elif attack_method == "AutoAttack":
            attacker = AutoAttack(surrogate_model, norm='Linf', eps=eps)
            attack_total = 0
            attack_correct = 0
            for ii, (data, label) in tqdm(enumerate(eval_dl)):
                data = data.type(torch.FloatTensor)
                data = data.to(device)
                label = label.to(device)
                if device != 'cpu':
                    perturbed_data = attacker(data, label).cuda(device=device)
                else:
                    perturbed_data = attacker(data, label)

                score = model(perturbed_data)
                _, predicted = torch.max(score, 1)
                attack_total += label.cpu().size(0)
                attack_correct += (predicted == label).sum()
        else:
            raise NameError("Unsupported Attack Method!")

        print(f'The robust accuracy against epsilon {eps} is {attack_correct / attack_total * 100}')
        result_list.append(attack_correct.cpu().item() / 100.)

    csv_row_list.append(result_list)
    write_csv_rows(file_path, csv_row_list)


if __name__ == '__main__':
    model_path_list = list(args.model_path.split(","))
    for model_path in model_path_list:
        evaluation(model_path)
