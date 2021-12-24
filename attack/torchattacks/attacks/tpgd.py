import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attack import Attack


class TPGD(Attack):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=7):
        super(TPGD, self).__init__("TPGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self._attack_mode = 'only_default'

    def forward(self, images, labels=None):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        logit_ori = self.model(images).detach()
        
        adv_images = images + 0.001*torch.randn_like(images)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        loss = nn.KLDivLoss(reduction='sum')

        for i in range(self.steps):
            adv_images.requires_grad = True
            logit_adv = self.model(adv_images)

            cost = loss(F.log_softmax(logit_adv, dim=1),
                        F.softmax(logit_ori, dim=1))

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
