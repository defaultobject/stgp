""" Collection of standard Trainers """

import objax
import jax

from .trainer import Trainer, ScipyTrainer, GradDescentTrainer, SwitchTrainer
from .natgrad_trainer import NatGradTrainer



class LBFGS(ScipyTrainer):
    """ Constructs a ScipyTrainer with L-BFGS-B argument """
    def __init__(self, *args, **kwargs):
        super(LBFGS, self).__init__(*args, optimizer='L-BFGS-B', **kwargs)

class VB_NG_ADAM(Trainer):
    """ 
    A variational bayes algorithm that uses Natural gradients to update the approximate posterior and Adam for the rest of the parameters.
    """
    def __init__(self, m, enforce_psd_type = None):
        self.ng_trainer = NatGradTrainer(m, enforce_psd_type = enforce_psd_type)

        # do not use adam to train the approximate posterior
        m.approximate_posterior.fix()
        self.adam_trainer = GradDescentTrainer(m, objax.optimizer.Adam)

        # construct switch trainer to iteratively update the above
        self.switch_trainer = SwitchTrainer(
            [self.adam_trainer, self.ng_trainer]
        )

    def train(
        self,
        learning_rates: list,
        epochs: list,
        callback = None

    ):
        """
        Args:
           learning_rates: list: [adam_lr, ng_lr] 
           epochs: list: [num_epochs, [adam_iters, ng_iters]] 
        """
        # first update the natural gradients, this is so the first gradients from adam are starting from a 'good point'
        self.ng_trainer.train(learning_rates[1], epochs[1][1])
        return self.switch_trainer.train(learning_rates, epochs, callback)

        
