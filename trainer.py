import torch
import numpy as np
import torch.nn.functional as F

from attack.pgd_attack import PgdAttack
from attack.pgd_attack_restart import attack_pgd_restart
from utils.context import ctx_noparamgrad
from utils.math_utils import l2_norm_batch as l2b


class BatTrainer:
    def __init__(self, args, log):
        self.args = args
        self.steps = args.attack_step
        self.eps = args.attack_eps
        self.attack_lr = args.attack_lr
        self.attack_rs = args.attack_rs
        self.lmbda = 1. / self.attack_lr
        if self.args.lmbda != 0.0:
            self.lmbda = self.args.lmbda

        self.constraint_type = np.inf if args.constraint_type == "linf" else 2
        self.log = log
        self.mode = args.mode
        self.sign_pgd = not args.pgd_no_sign
        self.z_init_non_sign_attack_lr = self.args.z_init_non_sign_attack_lr
        if self.z_init_non_sign_attack_lr is not None:
            self.z_init_non_sign_attack_lr /= 255

    def test_sa(self, model, data, labels):
        model.eval()
        with torch.no_grad():
            predictions_sa = model(data)
            correct = (torch.argmax(predictions_sa.data, 1) == labels).sum().cpu().item()
        return correct

    def get_input_grad(self, model, X, y, eps, delta_init='none', backprop=True):

        if delta_init == 'none':
            delta = torch.zeros_like(X, requires_grad=True)
        elif delta_init == 'random_uniform':
            delta = torch.zeros_like(X).uniform_(-eps, eps).requires_grad_(True)
        elif delta_init == 'random_corner':
            delta = torch.zeros_like(X).uniform_(-eps, eps).requires_grad_(True)
            delta = eps * torch.sign(delta)
        else:
            raise ValueError('wrong delta init')

        output = model(X + delta)
        loss = F.cross_entropy(output, y)
        grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]
        if not backprop:
            grad, delta = grad.detach(), delta.detach()
        return grad

    def get_perturbation_init(self, model, x, y, eps, device, method, z_init_detach=True):
        z_init = torch.clamp(
            x + torch.FloatTensor(x.shape).uniform_(-eps, eps).to(device),
            min=0, max=1
        ) - x
        z_init.requires_grad_(True)

        retain_graph = not z_init_detach
        pgd_attack_lr = 1.25 * eps
        fgsm_attack_lr = eps

        if method == "random":
            z = z_init

        elif method == "fgsm":
            model.clear_grad()
            model.with_grad()
            z_init = torch.zeros_like(x).requires_grad_(True)
            attack_loss_first = self._attack_loss(model(x + z_init), y)
            grad_attack_loss_delta_first = \
                torch.autograd.grad(attack_loss_first, z_init, retain_graph=retain_graph, create_graph=retain_graph)[0]
            z = z_init - fgsm_attack_lr * torch.sign(grad_attack_loss_delta_first)
            z = torch.clamp(x + z, min=0, max=1) - x

        elif method == "pgd":
            model.clear_grad()
            model.with_grad()
            attack_loss_first = self._attack_loss(model(x + z_init), y)
            grad_attack_loss_delta_first = \
                torch.autograd.grad(attack_loss_first, z_init, retain_graph=retain_graph, create_graph=retain_graph)[0]
            z = z_init - pgd_attack_lr * torch.sign(grad_attack_loss_delta_first)
            z = torch.clamp(z, min=-eps, max=eps)
            z = torch.clamp(x + z, min=0, max=1) - x

        elif method == "ns-pgd":
            model.clear_grad()
            model.with_grad()
            attack_loss_first = self._attack_loss(model(x + z_init), y)
            grad_attack_loss_delta_first = \
                torch.autograd.grad(attack_loss_first, z_init, retain_graph=retain_graph, create_graph=retain_graph)[0]
            z = z_init - self.z_init_non_sign_attack_lr * grad_attack_loss_delta_first
            z = torch.clamp(z, min=-eps, max=eps)
            z = torch.clamp(x + z, min=0, max=1) - x

        elif method == "ns-gd":
            model.clear_grad()
            model.with_grad()
            attack_loss_first = self._attack_loss(model(x + z_init), y)
            grad_attack_loss_delta_first = \
                torch.autograd.grad(attack_loss_first, z_init, retain_graph=retain_graph, create_graph=retain_graph)[0]
            z = z_init - self.z_init_non_sign_attack_lr * grad_attack_loss_delta_first

        elif method == "ns-pgd-zero":
            z_init = torch.zeros_like(x).requires_grad_(True)
            model.clear_grad()
            model.with_grad()
            attack_loss_first = self._attack_loss(model(x + z_init), y)
            grad_attack_loss_delta_first = \
                torch.autograd.grad(attack_loss_first, z_init, retain_graph=retain_graph, create_graph=retain_graph)[0]
            z = z_init - self.z_init_non_sign_attack_lr * grad_attack_loss_delta_first
            z = torch.clamp(z, min=-eps, max=eps)
            z = torch.clamp(x + z, min=0, max=1) - x

        else:
            raise NotImplementedError

        if z_init_detach:
            return z.detach()
        else:
            return z

    def train(self, model, train_dl, opt, loss_func, scheduler=None, device=None):
        if not device:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        adversary_train = PgdAttack(
            model, loss_fn=loss_func, eps=self.eps, steps=self.steps,
            eps_lr=self.attack_lr, ord=self.constraint_type,
            rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False,
            regular=0, sign=self.sign_pgd
        )

        model.train()
        one_count = torch.tensor([0.])
        one_total = torch.tensor([0.])
        cos_sim = torch.tensor([0.])
        p_avg_norm = torch.tensor([0.])
        cross_avg_norm = torch.tensor([0.])
        training_loss = torch.tensor([0.])
        train_sa = torch.tensor([0.])
        train_ra = torch.tensor([0.])
        verbose_flag = True

        total = 0

        for i, (data, labels) in enumerate(train_dl):
            data = data.type(torch.FloatTensor)
            data = data.to(device)
            labels = labels.to(device)
            real_batch = data.shape[0]
            channels = data.shape[1]
            image_size = data.shape[2]

            total += real_batch

            if self.mode == "fast_at":
                # Record SA along with each batch
                train_sa += self.test_sa(model, data, labels)

                model.train()
                if self.steps == 0:
                    delta_star = torch.zeros_like(data).to(data)
                else:

                    model.train()
                    opt.zero_grad()

                    delta_init = self.get_perturbation_init(model, data, labels, self.eps, device, self.args.z_method,
                                                            not self.args.z_no_detach)

                    with ctx_noparamgrad(model):
                        delta_star = adversary_train.perturb(data, labels, delta_init=delta_init) - data

                delta_star.requires_grad = False

                # Update model with perturbed data
                model.clear_grad()
                model.with_grad()
                predictions = model(data + delta_star)
                train_loss = loss_func(predictions, labels) / real_batch
                train_loss.backward()

                if self.args.verbose and verbose_flag:
                    with torch.no_grad():
                        p_tensor = []
                        for p in model.parameters():
                            p_tensor += p.grad.view(-1).tolist()

                        p_tensor = torch.tensor(p_tensor)
                        p_avg_norm = norm(p_tensor) / p_tensor.numel()
                        verbose_flag = False

                opt.step()

            elif self.mode == "pgd":
                # Record SA along with each batch
                train_sa += self.test_sa(model, data, labels)

                model.train()
                if self.steps == 0:
                    delta_star = torch.zeros_like(data).to(data)
                else:
                    model.train()
                    opt.zero_grad()

                    delta_init = self.get_perturbation_init(
                        model=model, x=data, y=labels, eps=self.eps, device=device, method=self.args.z_method,
                        z_init_detach=not self.args.z_no_detach
                    )

                    with ctx_noparamgrad(model):
                        delta_star = adversary_train.perturb(data, labels, delta_init=delta_init) - data

                delta_star.requires_grad = False

                # Update model with perturbed data
                model.clear_grad()
                model.with_grad()
                predictions = model(data + delta_star)
                train_loss = loss_func(predictions, labels) / real_batch
                train_loss.backward()
                opt.step()

            elif self.mode == "fast_at_ga":
                # Record SA along with each batch
                train_sa += self.test_sa(model, data, labels)

                model.train()

                double_bp = True if self.args.ga_coef > 0 else False

                X, y = data.to(device), labels.to(device)
                delta = torch.zeros_like(X, requires_grad=True)

                X_adv = torch.clamp(X + delta, 0, 1)
                output = model(X_adv)
                loss = F.cross_entropy(output, y)
                grad = torch.autograd.grad(loss, delta, create_graph=True if double_bp else False)[0]

                grad = grad.detach()

                argmax_delta = self.eps * torch.sign(grad)

                fgsm_alpha = 1.25
                delta.data = torch.clamp(delta.data + fgsm_alpha * argmax_delta, -self.eps, self.eps)
                delta.data = torch.clamp(X + delta.data, 0, 1) - X

                delta = delta.detach()

                predictions = model(X + delta)
                loss_function = torch.nn.CrossEntropyLoss()
                train_loss = loss_function(predictions, y)

                reg = self.get_ga_reg(model, data, labels, device, double_bp)

                train_loss += reg

                opt.zero_grad()
                train_loss.backward()

                if self.args.verbose and verbose_flag:
                    with torch.no_grad():
                        p_tensor = []
                        for p in model.parameters():
                            p_tensor += p.grad.view(-1).tolist()

                        p_tensor = torch.tensor(p_tensor)
                        p_avg_norm = norm(p_tensor) / p_tensor.numel()
                        verbose_flag = False

                opt.step()

            elif self.mode == "fast_bat":

                # Record SA along with each batch
                train_sa += self.test_sa(model, data, labels)

                model.train()

                z_init = torch.clamp(
                    data + torch.FloatTensor(data.shape).uniform_(-self.eps, self.eps).to(device),
                    min=0, max=1
                ) - data
                z_init.requires_grad_(True)

                model.clear_grad()
                model.with_grad()
                attack_loss = self._attack_loss(model(data + z_init), labels)
                grad_attack_loss_delta = torch.autograd.grad(attack_loss, z_init, retain_graph=True, create_graph=True)[
                    0]
                delta = z_init - self.attack_lr * grad_attack_loss_delta
                delta = torch.clamp(delta, min=-self.eps, max=self.eps)
                delta = torch.clamp(data + delta, min=0, max=1) - data

                delta = delta.detach().requires_grad_(True)
                attack_loss_second = self._attack_loss(model(data + delta), labels)
                grad_attack_loss_delta_second = \
                    torch.autograd.grad(attack_loss_second, delta, retain_graph=True, create_graph=True)[0] \
                        .view(real_batch, 1, channels * image_size * image_size)
                delta_star = delta - self.attack_lr * grad_attack_loss_delta_second.detach().view(data.shape)
                delta_star = torch.clamp(delta_star, min=-self.eps, max=self.eps)
                delta_star = torch.clamp(data + delta_star, min=0, max=1) - data
                z = delta_star.clone().detach().view(real_batch, -1)

                if self.constraint_type == np.inf:
                    # H: (batch, channel*image_size*image_size)
                    z_min = torch.max(-data.view(real_batch, -1),
                                      -self.eps * torch.ones_like(data.view(real_batch, -1)))
                    z_max = torch.min(1 - data.view(real_batch, -1),
                                      self.eps * torch.ones_like(data.view(real_batch, -1)))
                    H = ((z > z_min + 1e-7) & (z < z_max - 1e-7)).to(torch.float32)
                else:
                    raise NotImplementedError
                if self.args.verbose:
                    one_count += H.sum().cpu()
                    one_total += H.numel()

                delta_cur = delta_star.detach().requires_grad_(True)

                model.no_grad()
                lgt = model(data + delta_cur)
                delta_star_loss = loss_func(lgt, labels)
                delta_star_loss.backward()
                delta_outer_grad = delta_cur.grad.view(real_batch, -1)

                hessian_inv_prod = delta_outer_grad / self.lmbda
                bU = (H * hessian_inv_prod).unsqueeze(-1)

                model.with_grad()
                model.clear_grad()
                b_dot_product = grad_attack_loss_delta_second.bmm(bU).view(-1).sum(dim=0)
                b_dot_product.backward()
                cross_term = [-param.grad / real_batch for param in model.parameters()]

                model.clear_grad()
                model.with_grad()
                predictions = model(data + delta_star)
                train_loss = loss_func(predictions, labels) / real_batch
                opt.zero_grad()
                train_loss.backward()

                with torch.no_grad():
                    p_tensor = []
                    cross_tensor = []
                    for p, cross in zip(model.parameters(), cross_term):
                        new_grad = p.grad + cross

                        if verbose_flag and self.args.verbose:
                            p_tensor += p.grad.view(-1).tolist()
                            cross_tensor += cross.view(-1).tolist()

                        p.grad.copy_(new_grad)
                    if verbose_flag and self.args.verbose:
                        p_tensor = torch.tensor(p_tensor)
                        cross_tensor = torch.tensor(cross_tensor)
                        cos_sim = torch.sum(p_tensor * cross_tensor) / (norm(p_tensor) * norm(cross_tensor))
                        p_avg_norm = norm(p_tensor) / p_tensor.numel()
                        cross_avg_norm = norm(cross_tensor) / cross_tensor.numel()
                        verbose_flag = False
                del cross_term, H, grad_attack_loss_delta_second
                opt.step()

            else:
                raise NotImplementedError()

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == labels
                if self.log is not None:
                    self.log(model,
                             loss=train_loss.cpu(),
                             accuracy=correct.cpu(),
                             learning_rate=scheduler.get_last_lr()[0],
                             batch_size=real_batch)
            if scheduler:
                scheduler.step()

            training_loss += train_loss.cpu().sum().item()
            train_ra += correct.cpu().sum().item()
        return model, ((one_count / (one_total + 1e-7)).cpu().item(), cos_sim.cpu().item(), p_avg_norm.cpu().item(),
                       cross_avg_norm.cpu().item(), training_loss.cpu().item() / total, train_sa.cpu().item() / total,
                       train_ra.cpu().item() / total)

    def get_ga_reg(self, model, data, labels, device, double_bp):
        # Regularization for Gradient Alignment
        reg = torch.zeros(1).to(device)[0]
        delta = torch.zeros_like(data, requires_grad=True)
        output = model(torch.clamp(data + delta, 0, 1))
        clean_train_loss = F.cross_entropy(output, labels)
        grad = torch.autograd.grad(clean_train_loss, delta, create_graph=True if double_bp else False)[0]
        grad = grad.detach()

        if self.args.ga_coef != 0.0:
            grad_random_perturb = self.get_input_grad(model, data, labels, self.eps,
                                                      delta_init='random_uniform',
                                                      backprop=True)
            grads_nnz_idx = ((grad ** 2).sum([1, 2, 3]) ** 0.5 != 0) * (
                    (grad_random_perturb ** 2).sum([1, 2, 3]) ** 0.5 != 0)
            grad_clean_data, grad_random_perturb = grad[grads_nnz_idx], grad_random_perturb[grads_nnz_idx]
            grad_clean_data_norms, grad_random_perturb_norms = l2b(grad_clean_data), l2b(
                grad_random_perturb)
            grad_clean_data_normalized = grad_clean_data / grad_clean_data_norms[:, None, None, None]
            grad_random_perturb_normalized = grad_random_perturb / grad_random_perturb_norms[:, None, None,
                                                                   None]
            cos = torch.sum(grad_clean_data_normalized * grad_random_perturb_normalized, (1, 2, 3))
            reg += self.args.ga_coef * (1.0 - cos.mean())

        return reg

    def _attack_loss(self, predictions, labels):
        return -torch.nn.CrossEntropyLoss(reduction='sum')(predictions, labels)

    def eval(self, model, test_dl, attack_eps, attack_steps, attack_lr, attack_rs, device=None):
        total = 0
        robust_total = 0
        correct_total = 0
        test_loss = 0
        if not device:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        for ii, (data, labels) in enumerate(test_dl):
            data = data.type(torch.FloatTensor)
            data = data.to(device)
            labels = labels.to(device)
            real_batch = data.shape[0]
            total += real_batch

            with ctx_noparamgrad(model):
                perturbed_data = attack_pgd_restart(
                    model=model,
                    X=data,
                    y=labels,
                    eps=attack_eps,
                    alpha=attack_lr,
                    attack_iters=attack_steps,
                    n_restarts=attack_rs,
                    rs=(attack_rs > 1),
                    verbose=False,
                    linf_proj=True,
                    l2_proj=False,
                    l2_grad_update=False,
                    cuda=True
                ) + data

            if attack_steps == 0:
                perturbed_data = data

            predictions = model(data)
            correct = torch.argmax(predictions, 1) == labels
            correct_total += correct.sum().cpu().item()

            predictions = model(perturbed_data)
            robust = torch.argmax(predictions, 1) == labels
            robust_total += robust.sum().cpu().item()

            robust_loss = torch.nn.CrossEntropyLoss()(predictions, labels)
            test_loss += robust_loss.cpu().sum().item()

            if self.log:
                self.log(model=model,
                         accuracy=correct.cpu(),
                         robustness=robust.cpu(),
                         batch_size=real_batch)

        return correct_total, robust_total, total, test_loss / total


def norm(x):
    return torch.sqrt(torch.sum(x * x))
