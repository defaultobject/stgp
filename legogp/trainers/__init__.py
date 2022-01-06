from . trainer import Trainer, SimpleTrainer, ScipyTrainer, GradDescentTrainer, SwitchTrainer

from .natgrad_trainer import NatGradTrainer

__all__ = [
    'Trainer', 
    'SimpleTrainer',
    'ScipyTrainer',
    'NatGradTrainer',
    'GradDescentTrainer',
    'SwitchTrainer'
]
