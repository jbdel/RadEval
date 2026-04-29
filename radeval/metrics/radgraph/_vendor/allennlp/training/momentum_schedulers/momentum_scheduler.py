import torch

from RadEval.metrics.radgraph._vendor.allennlp.common.registrable import Registrable
from RadEval.metrics.radgraph._vendor.allennlp.training.scheduler import Scheduler


class MomentumScheduler(Scheduler, Registrable):
    def __init__(self, optimizer: torch.optim.Optimizer, last_epoch: int = -1) -> None:
        super().__init__(optimizer, "momentum", last_epoch)

    def get_values(self) -> None:
        raise NotImplementedError
