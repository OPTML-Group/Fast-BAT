import math
import torch
import matplotlib.pyplot as plt

class StepLR:
    def __init__(self, optimizer, learning_rate: float, total_epochs: int):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base = learning_rate

    def __call__(self, epoch):
        if epoch < self.total_epochs * 1/2:
            lr = self.base
        elif epoch < self.total_epochs * 3/4:
            lr = self.base * 0.1 ** 1
        else:
            lr = self.base * 0.1 ** 2

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class MultiLinearScheduler:
    def __init__(self, optimizer, lr_max, lr_middle, lr_min, x1, x2, x3):
        self.count = 0
        self.optimizer = optimizer
        self.lr = 0
        self.lr_max = lr_max
        self.lr_middle = lr_middle
        self.lr_min = lr_min
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self._set_lr()

    def step(self):
        self.count = self.count + 1
        self._calculate_cur_lr()
        self._set_lr()

    def get_last_lr(self):
        return [self.lr]

    def _set_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr

    def _calculate_cur_lr(self):
        if self.count < self.x1:
            self.lr = (self.lr_max - self.lr_min) / self.x1 * self.count
        elif self.count < self.x2:
            self.lr = self.lr_max - (self.lr_max - self.lr_middle) / (self.x2 - self.x1) * (self.count - self.x1)
        else:
            self.lr = self.lr_middle - (self.lr_middle - self.lr_min) / (self.x3 - self.x2) * (self.count - self.x2)


class CyclicLinQuaStepLR:
    def __init__(self, optimizer, lr_max, step_size_up, step_size_down):
        self.count = 0
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.step_total_up = step_size_up
        self.step_total_down = step_size_down
        self.lr = 0
        self._set_lr()

    def step(self):
        self.count = self.count + 1
        self._calculate_cur_lr()
        self._set_lr()

    def get_last_lr(self):
        return [self.lr]

    def _calculate_cur_lr(self):
        if self.count <= self.step_total_up:
            self.lr = self.lr_max * self.count / self.step_total_up
        else:
            self.lr = self.lr_max * ((self.count - self.step_total_up - self.step_total_down) ** 2) / (
                    self.step_total_down ** 2)

    def _set_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr


class CyclicLinBiQuaStepLR:
    def __init__(self, optimizer, lr_max, step_size_up, step_size_down):
        self.count = 0
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.step_total_up = step_size_up
        self.step_total_down = step_size_down
        self.lr = 0
        self._set_lr()

    def step(self):
        self.count = self.count + 1
        self._calculate_cur_lr()
        self._set_lr()

    def get_last_lr(self):
        return [self.lr]

    def _calculate_cur_lr(self):
        if self.count <= self.step_total_up:
            self.lr = self.lr_max * self.count / self.step_total_up
        else:
            self.lr = self.lr_max * ((self.count - self.step_total_up - self.step_total_down) ** 4) / (
                    self.step_total_down ** 4)

    def _set_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr


class CyclicQuadraticStepLR:
    def __init__(self, optimizer, lr_max, step_size_up, step_size_down):
        self.count = 0
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.step_total_up = step_size_up
        self.step_total_down = step_size_down
        self.lr = 0
        self._set_lr()

    def step(self):
        self.count = self.count + 1
        self._calculate_cur_lr()
        self._set_lr()

    def get_last_lr(self):
        return [self.lr]

    def _calculate_cur_lr(self):
        if self.count <= self.step_total_up:
            self.lr = self.lr_max * (self.count ** 2) / (self.step_total_up ** 2)
        else:
            self.lr = self.lr_max * ((self.count - self.step_total_up - self.step_total_down) ** 2) / ((self.step_total_down) ** 2)

    def _set_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr


class CosineAnnealingWarmupRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def get_last_lr(self):
        return self.get_lr()

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


if __name__ == "__main__":
    epochs = 50
    len = 200


    mile_stone = 5 * len
    mile_stone_mid = 10 * len
    mile_stone_final = epochs * len

    lr_max = 0.2
    lr_mid = 0.1

    x = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    for epoch in range(epochs):
        for pt in range(len):
            cur = epoch * len + pt
            x.append(epoch * len + pt)

            if cur < mile_stone:
                y1.append(lr_max * cur / mile_stone)
                y2.append(lr_max * cur / mile_stone)
                y3.append(lr_max * cur ** 2 / mile_stone ** 2)
                y4.append(lr_max * cur / mile_stone)
            elif cur < mile_stone_mid:
                y1.append(lr_max - (lr_max - lr_mid) / (mile_stone_mid - mile_stone) * (cur - mile_stone))
            else:
                y1.append(lr_mid - (lr_mid - 0) / (mile_stone_final - mile_stone_mid) * (cur - mile_stone_mid))

            if cur >= mile_stone:
                y2.append(lr_max * ((cur - mile_stone_final) ** 2) / (
                        (mile_stone_final - mile_stone) ** 2))
                y3.append(lr_max * ((cur - mile_stone_final) ** 2) / (
                        (mile_stone_final - mile_stone) ** 2))
                y4.append(lr_max * ((mile_stone_final - cur) ** 4) / (
                        (mile_stone_final - mile_stone) ** 4))
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.plot(x, y4)
    plt.show()