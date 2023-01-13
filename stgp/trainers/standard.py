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

class LikNoiseSplitTrainer(Trainer):
    """ A trainer that holds the likelihood noise for a percentage of the training epochs """
    def __init__(self, m, trainer_wrapper, hold_noise_percent):
        m.likelihood.fix()
        self.trainer_with_lik_held = trainer_wrapper()
        m.likelihood.release()
        self.trainer_with_lik_released = trainer_wrapper()
        self.hold_percent = hold_noise_percent
        
    def train(
        self,
        learning_rates: list,
        epochs: list,
        callback = None,
        verbose = False
    ):
        """
        Args:
            epochs: list: [num_epochs, *] 
        """

        # TODO: ensure valid values here
        max_iters = epochs[0]
        iters_with_lik_held = int(self.hold_percent * max_iters)
        iters_with_lik_released = max_iters -  iters_with_lik_held


        if verbose:
            # wrap with empty prints to add new lines to helping viewing when using callbacks
            print('')
            print(f'training with likelihood held for {iters_with_lik_held}/{max_iters}')
            print('')

        lc_1, _ = self.trainer_with_lik_held.train(learning_rates, [iters_with_lik_held, epochs[1]], callback)

        if verbose:
            print('')
            print(f'training with likelihood released for {iters_with_lik_released}/{max_iters}')
            print('')

        lc_2, _ = self.trainer_with_lik_released.train(learning_rates, [iters_with_lik_released, epochs[1]], callback)

        return [lc_1, lc_2], None
