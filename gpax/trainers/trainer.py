import objax

from timeit import default_timer as timer

import json

class Trainer:
    pass

class SimpleTrainer(Trainer):
    def summary(self, train_vars):
        print(train_vars)

    def train(self, m, optimizer, learning_rate, epochs, callback=None):
        train_vars = m.vars()

        self.summary(train_vars)

        objective_fn = objax.Jit(m.get_objective, train_vars)
        grad_fn = objax.Jit(objax.GradValues(objective_fn, train_vars), train_vars)
        #grad_fn = objax.GradValues(m.get_objective, train_vars)

        def train_op():
            grad, val = grad_fn()
            opt(learning_rate, grad)
            return grad, val

        opt = optimizer(train_vars)

        start = timer()

        epoch_arr = []

        for i in range(epochs):
            grad, val = train_op()

            if callback is not None:
                callback(i, grad, val)

            epoch_arr.append(val)

        end = timer()
        training_time = end - start

        return epoch_arr, training_time
