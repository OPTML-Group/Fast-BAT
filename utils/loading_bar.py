import time


class LoadingBar:
    def __init__(self, length: int = 40):
        self.length = length
        self.symbols = ['┈', '░', '▒', '▓']

    def __call__(self, progress: float) -> str:
        p = int(progress * self.length*4 + 0.5)
        d, r = p // 4, p % 4
        return '┠┈' + d * '█' + ((self.symbols[r]) + max(0, self.length-1-d) * '┈' if p < self.length*4 else '') + "┈┨"


class Log:
    def __init__(self, log_each: int, initial_epoch=-1):
        self.loading_bar = LoadingBar(length=27)
        self.best_accuracy = 0.0
        self.best_robustness = 0.0
        self.log_each = log_each
        self.epoch = initial_epoch

    def train(self, len_dataset: int) -> None:
        self.epoch += 1
        if self.epoch == 0:
            self._print_header()
        else:
            self.flush()

        self.is_train = True
        self.last_steps_state = {"loss_pgd": 0.0, "accuracy": 0.0, "steps": 0}
        self._reset(len_dataset)

    def eval(self, len_dataset: int) -> None:
        self.flush()
        self.is_train = False
        self._reset(len_dataset)

    def __call__(self, model, accuracy, loss=None, robustness=None, batch_size=None, learning_rate: float = None) -> None:
        if self.is_train:
            assert(loss is not None and learning_rate is not None)
            self._train_step(model, loss, accuracy, learning_rate, batch_size)
        else:
            assert(robustness is not None)
            self._eval_step(accuracy, robustness, batch_size)

    def flush(self) -> None:
        if self.is_train:
            loss = self.epoch_state["loss_pgd"] / self.epoch_state["steps"]
            accuracy = self.epoch_state["accuracy"] / self.epoch_state["steps"]
            print(
                f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{100*accuracy:10.2f} %  ┃{self.learning_rate:12.3e}  │{self._time():>12}  ┃",
                end="",
                flush=True,
            )

        else:
            accuracy = self.epoch_state["accuracy"] / self.epoch_state["steps"]
            robustness = self.epoch_state["robustness"] / self.epoch_state["steps"]
            print(f"{100*accuracy:10.2f} %  │{100*robustness:10.2f} %  ┃", flush=True)

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
            if robustness > self.best_robustness:
                self.best_robustness = robustness

    def _train_step(self, model, loss, accuracy, learning_rate: float, batch_size) -> None:
        self.learning_rate = learning_rate
        self.last_steps_state["loss_pgd"] += loss.sum().item() * batch_size
        self.last_steps_state["accuracy"] += accuracy.sum().item()
        self.last_steps_state["steps"] += batch_size
        self.epoch_state["loss_pgd"] += loss.sum().item() * batch_size
        self.epoch_state["accuracy"] += accuracy.sum().item()
        self.epoch_state["steps"] += batch_size
        self.step += 1

        if self.step % self.log_each == self.log_each - 1:
            loss = self.last_steps_state["loss_pgd"] / self.last_steps_state["steps"]
            accuracy = self.last_steps_state["accuracy"] / self.last_steps_state["steps"]

            self.last_steps_state = {"loss_pgd": 0.0, "accuracy": 0.0, "steps": 0}
            progress = self.step / self.len_dataset

            print(
                f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{100*accuracy:10.2f} %  ┃{learning_rate:12.3e}  │{self._time():>12}  {self.loading_bar(progress)}",
                end="",
                flush=True,
            )

    def _eval_step(self, accuracy, robustness, batch_size) -> None:
        self.epoch_state["robustness"] += robustness.sum().item()
        self.epoch_state["accuracy"] += accuracy.sum().item()
        self.epoch_state["steps"] += batch_size

    def _reset(self, len_dataset: int) -> None:
        self.start_time = time.time()
        self.step = 0
        self.len_dataset = len_dataset
        if self.is_train:
            self.epoch_state = {"loss_pgd": 0.0, "accuracy": 0.0, "steps": 0}
        else:
            self.epoch_state = {"accuracy": 0.0, "robustness": 0.0, "steps": 0}

    def _time(self) -> str:
        time_seconds = int(time.time() - self.start_time)
        return f"{time_seconds // 60:02d}:{time_seconds % 60:02d} min"

    def _print_header(self) -> None:
        print(f"┏━━━━━━━━━━━━━━┳━━━━━━━╸T╺╸R╺╸A╺╸I╺╸N╺━━━━━━━┳━━━━━━━╸S╺╸T╺╸A╺╸T╺╸S╺━━━━━━━┳━━━━━━━╸V╺╸A╺╸L╺╸I╺╸D╺━━━━━━━┓")
        print(f"┃              ┃              ╷              ┃              ╷              ┃              ╷              ┃")
        print(f"┃       epoch  ┃    loss_pgd  │    accuracy  ┃        l.r.  │     elapsed  ┃   accuracy   │  robustness  ┃")
        print(f"┠──────────────╂──────────────┼──────────────╂──────────────┼──────────────╂──────────────┼──────────────┨")
