import torch
import numpy as np
from tqdm import tqdm
import seaborn as sns
from advertorch.attacks import LinfPGDAttack
from advertorch.context import ctx_noparamgrad
import matplotlib.pyplot as plt

sns.set()


def plot_adv_freq(model, data_loader, output_path="output/plot/adv_freq_plot.png"):
    model.eval()

    total = 0
    robust_total = 0

    adversary_eval = LinfPGDAttack(
        model, loss_fn=torch.nn.CrossEntropyLoss(), eps=8 / 255, nb_iter=20, eps_iter=2 / 255,
        rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False
    )

    image_size = data_loader.dataset[0][0].shape[1]
    channel_num = data_loader.dataset[0][0].shape[0]
    f_sum = torch.zeros((image_size, image_size))

    for ii, (data, labels) in tqdm(enumerate(data_loader)):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        data = data.type(torch.FloatTensor)
        data = data.to(device)
        labels = labels.to(device)
        real_batch = data.shape[0]
        total += real_batch

        with ctx_noparamgrad(model):
            perturbed_data = adversary_eval.perturb(data, labels)
            perturbation = perturbed_data - data

        predictions = model(perturbed_data)
        robust = torch.argmax(predictions, 1) == labels
        robust_total += robust.sum().cpu().item()

        # Only attack success image
        for i, res in enumerate(robust):

            if res:
                continue

            f_single = torch.zeros((image_size, image_size))
            for ch in range(channel_num):
                f_perturb = np.abs(np.fft.fftshift(np.fft.fft2(perturbation.detach().cpu()[i][ch, :, :]))) / channel_num
                f_single += f_perturb

            f_sum += f_single

    f_sum = f_sum / (total - robust_total)
    plt.figure()
    sns_plot = sns.heatmap(f_sum, vmax=6.0, vmin=0.0)
    plt.savefig(output_path)
    plt.close()
