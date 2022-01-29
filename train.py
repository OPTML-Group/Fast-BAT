import argparse
import logging
import os

from datasets import *
from model_zoo import *
from trainer import BatTrainer
from utils.general_utils import write_csv_rows, setup_seed
from utils.lamb import Lamb
from utils.loading_bar import Log
from utils.math_utils import smooth_crossentropy, dlr_loss
from utils.step_lr import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Bi-level Adversarial Training")

    ########################## basic setting ##########################
    parser.add_argument('--device', default="cuda:0", help="The name of the device you want to use (default: cuda:0)")
    parser.add_argument('--time_stamp', default="debug",
                        help="The time stamp that helps identify different trails.")
    parser.add_argument('--dataset', default="CIFAR10",
                        choices=["CIFAR10", "CIFAR100", "TINY_IMAGENET", "IMAGENET", "SVHN"])
    parser.add_argument('--dataset_val_ratio', default=0.0, type=float)
    parser.add_argument('--mode', default='fast_bat', type=str,
                        choices=["fast_at", "fast_bat", "fast_at_ga", "pgd"],
                        help="fast-at : pgd-at, fast_bat_kkt: bi-level at with kkt, fast_at_ga: gradient alignment")
    parser.add_argument('--data_dir', default='./data/', type=str, help="The folder where you store your dataset")
    parser.add_argument('--model_prefix', default='checkpoints/',
                        help='File folders where you want to store your checkpoints (default: results/checkpoints/)')
    parser.add_argument('--csv_prefix', default='accuracy/',
                        help='File folders where you want to put your results (default: results/accruacy)')
    parser.add_argument('--random_seed', default=37, type=int,
                        help='Random seed (default: 37)')
    parser.add_argument('--pretrained_model', default=None, help="The path of pretrained model")
    parser.add_argument('--pretrained_epochs', default=0, type=int)

    ########################## training setting ##########################
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument("--epochs", default=20, type=int, help="Total number of epochs.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--optimizer", default="SGD", choices=['SGD', 'Adam', 'Lamb'])
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate.")

    ########################## learning scheduler ##########################
    parser.add_argument('--lr_scheduler', default='cyclic',
                        choices=['cyclic', 'multistep'])
    parser.add_argument("--cyclic_milestone", default=10, type=int)
    parser.add_argument("--key_epochs", nargs="+", type=int, default=[100, 150],
                        help="Epochs where learning rate decays, this is for multi-step scheduler only.")
    parser.add_argument("--lr_decay_rate", default=0.1, type=float, help="This is for multi-step scheduler only.")
    parser.add_argument('--lr_min', default=0., type=float)
    parser.add_argument('--lr_max', default=0.2, type=float)

    ########################## model setting ##########################
    parser.add_argument('--train_loss', default="ce", choices=["ce", "sce", "n_dlr"],
                        help="ce for cross entropy, sce for label-smoothed ce, n_dlr for negative dlr loss")
    parser.add_argument('--act_fn', default="relu", choices=["relu", "softplus", "swish"],
                        help="choose the activation function for your model")
    parser.add_argument("--model_type", default="PreActResNet", choices=['ResNet', 'PreActResNet', 'WideResNet'])
    parser.add_argument("--width_factor", default=0, type=int, help="Parameter for WideResNet only.")
    parser.add_argument("--depth", default=18, type=int, help="Parameter for all model types.")

    ########################## attack setting ##########################
    parser.add_argument('--attack_step', default=1, type=int,
                        help='attack steps for training (default: 1)')
    parser.add_argument('--attack_step_test', default=50, type=int,
                        help='attack steps for evaluation (default: 50)')
    parser.add_argument('--attack_eps', default=8, type=float,
                        help='attack constraint for training (default: 8/255)')
    parser.add_argument('--attack_rs', default=1, type=int,
                        help='attack restart number')
    parser.add_argument('--attack_lr', default=2., type=float,
                        help='attack learning rate (default: 2./255). Note this parameter is for training only. The attack lr is always set to attack_eps / 4 when evaluating.')
    parser.add_argument('--attack_rs_test', default=10, type=int,
                        help='attack restart number for evaluation')

    ############################### fast-bat options ###################################
    parser.add_argument('--lmbda', default=10.0, type=float, help="The parameter lambda for Fast-BAT.")

    ############################### grad alignment ##################################
    parser.add_argument('--ga_coef', default=0.0, type=float,
                        help="coefficient of the cosine gradient alignment regularizer")

    args = parser.parse_args()
    device = args.device
    if device != "cpu" and torch.cuda.is_available():
        # Please use CUDA_VISIBLE_DEVICES to assign gpu
        device = "cuda:0"

    result_path = "./results/"
    log_path = "./log/"
    model_dir = os.path.join(result_path, args.model_prefix)
    csv_dir = os.path.join(result_path, args.csv_prefix)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(csv_dir):
        os.mkdir(csv_dir)

    if args.mode == "fast_at" or args.mode == "fast_at_ga":
        args.attack_lr = args.attack_eps * 1.25 / 255
    elif args.mode == "pgd":
        if args.attack_step == 2:
            args.attack_lr = args.attack_eps * 0.5 / 255
        elif args.attack_step == 10:
            args.attack_lr = 2.0 / 255
        else:
            args.attack_lr = args.attack_lr / 255
    else:
        if args.attack_eps <= 8:
            args.attack_lr = 5000
        else:
            args.attack_lr = 2000
        args.attack_lr = args.attack_lr / 255

    args.attack_eps = args.attack_eps / 255

    setup_seed(seed=args.random_seed)
    training_type = args.mode.upper()
    model_name = f"{args.dataset}_{training_type}_{args.time_stamp}"
    model_path = os.path.join(result_path, args.model_prefix + model_name + '.pth')
    best_model_path = os.path.join(result_path, args.model_prefix + model_name + '_best.pth')
    csv_path = os.path.join(result_path, args.csv_prefix + model_name + '.csv')

    ############################## Logger #################################
    log = Log(log_each=2)
    logging.basicConfig(filename=os.path.join(log_path, f'{model_name}.log'), level=logging.INFO)
    logger = logging.getLogger("CIFAR10 BAT Training")

    ########################## dataset and model ##########################
    if args.dataset == "CIFAR10":
        train_dl, val_dl, test_dl, norm_layer = cifar10_dataloader(data_dir=args.data_dir,
                                                                   batch_size=args.batch_size,
                                                                   val_ratio=args.dataset_val_ratio)
        num_classes = 10
        conv1_size = 3
    elif args.dataset == "CIFAR100":
        train_dl, val_dl, test_dl, norm_layer = cifar100_dataloader(data_dir=args.data_dir,
                                                                    batch_size=args.batch_size,
                                                                    val_ratio=args.dataset_val_ratio)
        num_classes = 100
        conv1_size = 3
    elif args.dataset == "IMAGENET":
        train_dl, val_dl, test_dl, norm_layer = imagenet_dataloader(data_dir=args.data_dir,
                                                                    batch_size=args.batch_size)
        num_classes = 1000
        conv1_size = 3

    elif args.dataset == "TINY_IMAGENET":
        train_dl, val_dl, test_dl, norm_layer = tiny_imagenet_dataloader(data_dir=args.data_dir,
                                                                         batch_size=args.batch_size)
        num_classes = 200
        conv1_size = 3

    elif args.dataset == "SVHN":
        train_dl, val_dl, test_dl, norm_layer = svhn_dataloader(data_dir=args.data_dir,
                                                                batch_size=args.batch_size)
        num_classes = 10
        conv1_size = 3
    else:
        raise NotImplementedError("Invalid Dataset")

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
    model = model.to(device)

    ########################## optimizer and scheduler ##########################
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay,
                                    momentum=args.momentum)
    elif args.optimizer == "Lamb":
        optimizer = Lamb(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay,
                         betas=(0.9, 0.999))
    else:
        raise NotImplementedError("Unsupported optimizer!")

    lr_steps = args.epochs * len(train_dl)
    if args.lr_scheduler == "cyclic":
        milestone_epoch_num = args.cyclic_milestone
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                      base_lr=args.lr_min,
                                                      max_lr=args.lr_max,
                                                      step_size_up=int(milestone_epoch_num * len(train_dl)),
                                                      step_size_down=int(
                                                          (args.epochs - milestone_epoch_num) * len(train_dl)))
    elif args.lr_scheduler == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[len(train_dl) * i for i in args.key_epochs],
                                                         gamma=args.lr_decay_rate)
    else:
        raise NotImplementedError("Unsupported Scheduler!")

    if args.train_loss == "sce":
        train_loss = smooth_crossentropy
    elif args.train_loss == "ce":
        train_loss = torch.nn.CrossEntropyLoss(reduction="sum")
    elif args.train_loss == "n_dlr":
        def n_dlr(predictions, labels):
            return -dlr_loss(predictions, labels)
        train_loss = n_dlr
    else:
        raise NotImplementedError("Unsupported Loss Function!")

    ############################ BAT Trainer ###################################
    trainer = BatTrainer(args=args,
                         log=log)

    ########################## resume ##########################
    if args.pretrained_model:
        model.load(args.pretrained_model, map_location=device)

    for epoch in range(0, args.pretrained_epochs):
        for i in range(len(train_dl)):
            optimizer.step()
            scheduler.step()

    param_info = f"training type: {training_type}\n" + \
                 f"device: {args.device}\n" + \
                 f"model name: {model_name}\n" + \
                 f"epoch number: {args.epochs}\n" + \
                 f"random seed: {args.random_seed}\n" + \
                 f"key epoch: {args.key_epochs}\n" + \
                 f"batch size: {args.batch_size}\n" + \
                 f"validation set ratio: {args.dataset_val_ratio}\n" + \
                 f"model type: {args.model_type}\n" + \
                 f"model depth: {args.depth}\n" + \
                 f"model width: {args.width_factor}\n" + \
                 f"scheduler: {args.lr_scheduler}\n" + \
                 f"learning rate decay rate for multi-step: {args.lr_decay_rate}\n" + \
                 f"max learning rate: {args.lr_max}\n" + \
                 f"weight_decay: {args.weight_decay}\n" + \
                 f"momentum: {args.momentum}\n" \
                 f"dropout: {args.dropout}\n" + \
                 f"attack learning rate: {args.attack_lr * 255} / 255\n" \
                 f"attack epsilon: {args.attack_eps * 255} / 255\n" \
                 f"attack step: {args.attack_step}\n" + \
                 f"attack restart: {args.attack_rs}\n" + \
                 f"evaluation attack step: {args.attack_step_test}\n" + \
                 f"evaluation attack restart: {args.attack_rs_test}\n" + \
                 f"pretrained model: {args.pretrained_model}\n" + \
                 f"pretrained epochs: {args.pretrained_epochs}\n" + \
                 f"lambda: {args.lmbda}\n" + \
                 f"gradient alignment cosine coefficient: {args.ga_coef}\n"

    logger.info(param_info)
    print(param_info)

    epoch_num_list = ['Epoch Number']
    training_sa_list = ['Training Standard Accuracy']
    training_ra_list = ['Training Robust Accuracy']
    test_sa_list = ['Test Standard Accuracy']
    test_ra_list = ['Test Robust Accuracy']
    training_loss_list = ['Training Loss']
    test_loss_list = ['Test Loss']

    best_acc = 0.0

    for epoch in range(args.pretrained_epochs, args.epochs):
        logger.info(f"\n========================Here z a New Epoch : {epoch}========================")
        model.train()
        csv_row_list = []
        log.train(len_dataset=len(train_dl))

        model = trainer.train(model=model,
                              train_dl=train_dl,
                              opt=optimizer,
                              loss_func=train_loss,
                              scheduler=scheduler,
                              device=device)

        model.eval()
        log.eval(len_dataset=len(test_dl))

        correct_total, robust_total, total, test_loss = trainer.eval(model=model,
                                                                     test_dl=test_dl,
                                                                     attack_eps=args.attack_eps,
                                                                     attack_steps=args.attack_step_test,
                                                                     attack_lr=args.attack_eps / 4,
                                                                     attack_rs=args.attack_rs_test,
                                                                     device=device)

        natural_acc = correct_total / total
        robust_acc = robust_total / total

        # Writing data into csv file
        epoch_num_list.append(epoch)
        csv_row_list.append(epoch_num_list)
        csv_row_list.append(training_loss_list)
        csv_row_list.append(training_sa_list)
        csv_row_list.append(training_ra_list)

        test_loss_list.append(test_loss)
        csv_row_list.append(test_loss_list)
        test_sa_list.append(100. * natural_acc)
        csv_row_list.append(test_sa_list)
        test_ra_list.append(100. * robust_acc)
        csv_row_list.append(test_ra_list)

        logger.info(f'For the epoch {epoch} the test loss is {test_loss}')
        logger.info(f'For the epoch {epoch} the standard accuracy is {natural_acc}')
        logger.info(f'For the epoch {epoch} the robust accuracy is {robust_acc}')

        model.save(model_path)
        write_csv_rows(csv_path, csv_row_list)

        if robust_acc > best_acc:
            best_acc = robust_acc
            model.save(best_model_path)

    log.flush()
    print('Training Over')
