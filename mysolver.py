from modulus.solver import Solver
from omegaconf import DictConfig
import warnings

from modulus.trainer import Trainer
from modulus.domain import Domain

class MySolver(Solver):
    def __init__(self, cfg: DictConfig, domain: Domain) -> None:
        super(MySolver, self).__init__(cfg,domain)
        pass



