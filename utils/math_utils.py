import torch
import numpy as np
from torch.distributions import laplace
from torch.distributions import uniform
import torch.nn.functional as F


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


def hessian_vec_prod(func, x, vec, grads=None):
    x.requires_grad = True
    if grads is None:
        inner_res = func(x)
        grads = torch.autograd.grad(inner_res, x, create_graph=True, retain_graph=True)[0].view(-1)
    prod = grads.dot(vec.double())
    so_grad = torch.autograd.grad(prod, x, retain_graph=True)[0].view(-1)
    return so_grad


def hessian_vec_prod_complex(func, x, vecs):
    def get_second_order_grad(_func, x):
        x.requires_grad = True

        inner_res = _func(x)
        grads = torch.autograd.grad(inner_res, x, create_graph=True, retain_graph=True)

        grads2 = torch.tensor([])
        for anygrad in grads[0]:
            grads2 = torch.cat((grads2, torch.autograd.grad(anygrad, x, retain_graph=True)[0]))
        return grads2.view(x.size()[0], -1)

    x.requires_grad = True
    grads_so = get_second_order_grad(func, x)
    prod = grads_so @ vecs.double()
    return prod


def hessian_vec_prod_diff(inner_func, x, y, vecs, r=1e-7):
    x.requires_grad = True
    y.requires_grad = True

    def get_grad(func, outer_var, inner_var):
        inner_res = func(outer_var, inner_var)
        grads_dy = torch.autograd.grad(inner_res
                                       , outer_var
                                       , create_graph=True
                                       , retain_graph=True
                                       )[0].view(-1)
        return grads_dy

    def add(inner_var, vec, omega):
        return inner_var + omega * vec

    y_right = add(y, vecs, r)
    y_left = add(y, vecs, -r)

    g_lefts = get_grad(inner_func, x, y_left)
    g_rights = get_grad(inner_func, x, y_right)

    return (g_rights - g_lefts) / (2 * r)


def hessian_cross_calculation(inner_func, x, y):
    x.requires_grad = True
    y.requires_grad = True
    res = inner_func(x, y)
    grads = torch.autograd.grad(res, y, create_graph=True, retain_graph=True)

    grads2 = torch.tensor([])
    for anygrad in grads[0]:
        grads2 = torch.cat((grads2, torch.autograd.grad(anygrad, x, retain_graph=True)[0]))
    return grads2.view(x.size()[0], -1)


def hessian_gaussian_estimation(func, x, mu=1e-2, sample=100, lam=1e-3):
    x = x.view(-1)
    d = x.size()[0]
    hessian = torch.zeros((x.size()[0], x.size()[0]))

    for i in range(sample):
        u = torch.randn(d)
        res = (func(x + mu * u) + func(x - mu * u) - 2 * func(x)) / (2 * mu ** 2)
        u = u.unsqueeze(-1)
        hessian = hessian + res * (u.matmul(u.t()) - torch.eye(d))

    hessian /= sample

    return hessian + lam * torch.eye(d)


def batch_dot(v1, v2):
    assert v1.shape == v2.shape
    batch_size = v1.shape[0]
    return v1.view(batch_size, 1, -1).bmm(v2.view(batch_size, -1, 1)).squeeze(-1)


def batch_cg_solver(fx, b, iter_num=1000, residual_tol=1e-7, x_init=None, verbose=False):
    x = torch.zeros(b.shape).float().to(b) if x_init is None else x_init

    r = b - fx(x)
    p = r.clone()

    for i in range(iter_num):
        rdotr = batch_dot(r, r)
        Ap = fx(p)
        alpha = rdotr / (batch_dot(p, Ap) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        newrdotr = batch_dot(r, r)
        if verbose:
            print(f"BG iteration {i}, {newrdotr.mean()}")

        if newrdotr.mean() < residual_tol:
            if verbose:
                print(f"Early CG termination at iteration {i}")
            break

        beta = newrdotr / (rdotr + 1e-12)
        p = r + beta * p

    return x


def cg_solver(fx, b, iter_num=10, residual_tol=1e-10, x_init=None):
    x = torch.zeros(b.shape[0]).double() if x_init is None else x_init
    if b.dtype == torch.float16:
        x = x.half()
    r = (b - fx(x))
    p = r.clone()

    for i in range(iter_num):
        rdotr = r.dot(r)
        Ap = fx(p)
        alpha = rdotr / (p.dot(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        newrdotr = r.dot(r)
        beta = newrdotr / rdotr
        p = r + beta * p

        if newrdotr < residual_tol:
            break
    return x


def dlr_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()

    return ((x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (
            x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)).mean()


def smooth_crossentropy(pred, gold, smoothing=0.3):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1).mean()


def l2_norm_batch(v):
    norms = (v ** 2).sum([1, 2, 3]) ** 0.5
    return norms


if __name__ == "__main__":
    AA = []
    for i in range(4):
        A = torch.rand((5, 5))
        AA.append(A.matmul(A.t()).unsqueeze(0))

    AA = torch.cat(AA, dim=0)

    b = torch.rand((4, 5))


    def __for_batch_cg(x):
        return AA.bmm(x.unsqueeze(-1)).squeeze(-1)


    res = batch_cg_solver(__for_batch_cg, b)

    res_2 = torch.inverse(AA).bmm(b.unsqueeze(-1)).squeeze(-1)
    print(res - res_2)
