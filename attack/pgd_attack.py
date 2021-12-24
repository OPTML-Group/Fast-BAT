import torch
from torch import nn
from torch.distributions import laplace
from torch.distributions import uniform

import numpy as np


class Attack(object):

    def __init__(self, predict, loss_fn, clip_min, clip_max):
        """Create an Attack instance."""
        self.predict = predict
        self.loss_fn = loss_fn
        self.clip_min = clip_min
        self.clip_max = clip_max

    def perturb(self, x, **kwargs):

        error = "Sub-classes must implement perturb."
        raise NotImplementedError(error)

    def __call__(self, *args, **kwargs):
        return self.perturb(*args, **kwargs)


class LabelMixin(object):
    def _get_predicted_label(self, x):

        with torch.no_grad():
            outputs = self.model(x)
        _, y = torch.max(outputs, dim=1)
        return y

    def _verify_and_process_inputs(self, x, y):
        if self.targeted:
            assert y is not None

        if not self.targeted:
            if y is None:
                y = self._get_predicted_label(x)

        x = x.detach().clone()
        y = y.detach().clone()
        return x, y


class PgdAttack(Attack, LabelMixin):
    def __init__(self, model, eps, steps, eps_lr, loss_fn=None, rand_init=True, clip_min=0.0, clip_max=1.0, ord=np.inf,
                 l1_sparsity=None, targeted=False, regular=0.0, sign=True):
        super(PgdAttack, self).__init__(
            model, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.steps = steps
        self.eps_lr = eps_lr
        self.loss_fn = loss_fn if not loss_fn else nn.CrossEntropyLoss(reduction="sum")
        self.rand_init = rand_init
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.model = model
        self.regular = regular
        self.targeted = targeted
        self.l1_sparsity = l1_sparsity
        self.ord = ord
        self.sign = sign
        self.delta_gross = None

        assert is_float_or_torch_tensor(self.eps_lr)
        assert is_float_or_torch_tensor(self.eps)

    def get_delta_gross(self):
        return self.delta_gross

    def perturb(self, x, y=None, delta_init=None):
        x, y = self._verify_and_process_inputs(x, y)

        if delta_init is None:

            delta = torch.zeros_like(x)
            delta = nn.Parameter(delta)

            if self.rand_init:
                rand_init_delta(
                    delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
                delta.data = clamp(
                    x + delta.data, min=self.clip_min, max=self.clip_max) - x
        else:
            delta = delta_init.detach().clone()

        rval = self.perturb_iterative(
            x, y, self.predict, nb_iter=self.steps,
            eps=self.eps, eps_iter=self.eps_lr,
            loss_fn=self.loss_fn, minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init=delta,
            l1_sparsity=self.l1_sparsity, reg=self.regular, sign=self.sign
        )

        return rval.data

    def perturb_iterative(self, xvar, yvar, predict, nb_iter, eps, eps_iter, loss_fn,
                          delta_init=None, minimize=False, ord=np.inf,
                          clip_min=0.0, clip_max=1.0,
                          l1_sparsity=None, reg=0.0, sign=True):

        if delta_init is not None:
            delta = delta_init
        else:
            delta = torch.zeros_like(xvar)

        delta.requires_grad_()
        for ii in range(nb_iter):
            outputs = predict(xvar + delta)
            loss = loss_fn(outputs, yvar)
            if minimize:
                loss = -loss

            loss = loss + reg * torch.sum(delta * delta)

            loss.backward()
            if ord == np.inf:
                if sign:
                    delta_grad = delta.grad.data.sign()
                else:
                    delta_grad = delta.grad.data
                delta.data = delta.data + batch_multiply(eps_iter, delta_grad)

                self.delta_gross = delta.clone().detach()

                delta.data = batch_clamp(eps, delta.data)
                delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                                   ) - xvar.data

            elif ord == 2:
                grad = delta.grad.data
                grad = normalize_by_pnorm(grad)
                delta.data = delta.data + batch_multiply(eps_iter, grad)
                delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                                   ) - xvar.data
                if eps is not None:
                    delta.data = clamp_by_pnorm(delta.data, ord, eps)

            elif ord == 1:
                grad = delta.grad.data
                abs_grad = torch.abs(grad)

                batch_size = grad.size(0)
                view = abs_grad.view(batch_size, -1)
                view_size = view.size(1)
                if l1_sparsity is None:
                    vals, idx = view.topk(1)
                else:
                    vals, idx = view.topk(
                        int(np.round((1 - l1_sparsity) * view_size)))

                out = torch.zeros_like(view).scatter_(1, idx, vals)
                out = out.view_as(grad)
                grad = grad.sign() * (out > 0).float()
                grad = normalize_by_pnorm(grad, p=1)
                delta.data = delta.data + batch_multiply(eps_iter, grad)

                delta.data = batch_l1_proj(delta.data.cpu(), eps)
                if xvar.is_cuda:
                    delta.data = delta.data.cuda()
                delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                                   ) - xvar.data
            else:
                error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
                raise NotImplementedError(error)
            delta.grad.data.zero_()

        x_adv = clamp(xvar + delta, clip_min, clip_max)
        return x_adv


def _thresh_by_magnitude(theta, x):
    return torch.relu(torch.abs(x) - theta) * x.sign()


def batch_l1_proj_flat(x, z=1):
    # Computing the l1 norm of v
    v = torch.abs(x)
    v = v.sum(dim=1)

    # Getting the elements to project in the batch
    indexes_b = torch.nonzero(v > z).view(-1)
    if isinstance(z, torch.Tensor):
        z = z[indexes_b][:, None]
    x_b = x[indexes_b]
    batch_size_b = x_b.size(0)

    # If all elements are in the l1-ball, return x
    if batch_size_b == 0:
        return x

    # make the projection on l1 ball for elements outside the ball
    view = x_b
    view_size = view.size(1)
    mu = view.abs().sort(1, descending=True)[0]
    vv = torch.arange(view_size).float().to(x.device)
    st = (mu.cumsum(1) - z) / (vv + 1)
    u = (mu - st) > 0
    if u.dtype.__str__() == "torch.bool":  # after and including torch 1.2
        rho = (~u).cumsum(dim=1).eq(0).sum(1) - 1
    else:  # before and including torch 1.1
        rho = (1 - u).cumsum(dim=1).eq(0).sum(1) - 1
    theta = st.gather(1, rho.unsqueeze(1))
    proj_x_b = _thresh_by_magnitude(theta, x_b)

    # gather all the projected batch
    proj_x = x.detach().clone()
    proj_x[indexes_b] = proj_x_b
    return proj_x


def batch_l1_proj(x, eps):
    batch_size = x.size(0)
    view = x.view(batch_size, -1)
    proj_flat = batch_l1_proj_flat(view, z=eps)
    return proj_flat.view_as(x)


def batch_clamp(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_clamp_tensor_by_vector(float_or_vector, tensor)
        return tensor
    elif isinstance(float_or_vector, float):
        tensor = clamp(tensor, -float_or_vector, float_or_vector)
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor


def _batch_clamp_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following
    for ii in range(len(vector)):
        batch_tensor[ii] = clamp(
            batch_tensor[ii], -vector[ii], vector[ii])
    """
    return torch.min(
        torch.max(batch_tensor.transpose(0, -1), -vector), vector
    ).transpose(0, -1).contiguous()


def rand_init_delta(delta, x, ord, eps, clip_min, clip_max):
    if isinstance(eps, torch.Tensor):
        assert len(eps) == len(delta)

    if ord == np.inf:
        delta.data.uniform_(-1, 1)
        delta.data = batch_multiply(eps, delta.data)
    elif ord == 2:
        delta.data.uniform_(clip_min, clip_max)
        delta.data = delta.data - x
        delta.data = clamp_by_pnorm(delta.data, ord, eps)
    elif ord == 1:
        ini = laplace.Laplace(
            loc=delta.new_tensor(0), scale=delta.new_tensor(1))
        delta.data = ini.sample(delta.data.shape)
        delta.data = normalize_by_pnorm(delta.data, p=1)
        ray = uniform.Uniform(0, eps).sample()
        delta.data *= ray
        delta.data = clamp(x.data + delta.data, clip_min, clip_max) - x.data
    else:
        error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
        raise NotImplementedError(error)

    delta.data = clamp(
        x + delta.data, min=clip_min, max=clip_max) - x
    return delta.data


def is_float_or_torch_tensor(x):
    return isinstance(x, torch.Tensor) or isinstance(x, float)


def batch_multiply(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_multiply_tensor_by_vector(float_or_vector, tensor)
    elif isinstance(float_or_vector, float):
        tensor *= float_or_vector
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor


def _batch_multiply_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following
    for ii in range(len(vector)):
        batch_tensor.data[ii] *= vector[ii]
    return batch_tensor
    """
    return (
            batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()


def _get_norm_batch(x, p):
    batch_size = x.size(0)
    return x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)


def clamp_by_pnorm(x, p, r):
    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    if isinstance(r, torch.Tensor):
        assert norm.size() == r.size()
    else:
        assert isinstance(r, float)
    factor = torch.min(r / norm, torch.ones_like(norm))
    return batch_multiply(factor, x)


def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    """
    Normalize gradients for gradient (not gradient sign) attacks.

    :param x: tensor containing the gradients on the input.
    :param p: (optional) order of the norm for the normalization (1 or 2).
    :param small_constant: (optional float) to avoid dividing by zero.
    :return: normalized gradients.
    """
    # loss is averaged over the batch so need to multiply the batch
    # size to find the actual gradient of each input sample

    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    norm = torch.max(norm, torch.ones_like(norm) * small_constant)
    return batch_multiply(1. / norm, x)


def clamp(input, min=None, max=None):
    ndim = input.ndimension()
    if min is None:
        pass
    elif isinstance(min, (float, int)):
        input = torch.clamp(input, min=min)
    elif isinstance(min, torch.Tensor):
        if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
            input = torch.max(input, min.view(1, *min.shape))
        else:
            assert min.shape == input.shape
            input = torch.max(input, min)
    else:
        raise ValueError("min can only be None | float | torch.Tensor")

    if max is None:
        pass
    elif isinstance(max, (float, int)):
        input = torch.clamp(input, max=max)
    elif isinstance(max, torch.Tensor):
        if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
            input = torch.min(input, max.view(1, *max.shape))
        else:
            assert max.shape == input.shape
            input = torch.min(input, max)
    else:
        raise ValueError("max can only be None | float | torch.Tensor")
    return input
