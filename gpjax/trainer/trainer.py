import objax

from timeit import default_timer as timer

class Trainer:
    pass

class SimpleTrainer(Trainer):
    def train(self, m, optimizer, learning_rate, epochs, callback=None):
        train_vars = m.vars()

        objective_fn = objax.Jit(m.get_objective, train_vars)
        grad_fn = objax.Jit(objax.GradValues(objective_fn, m.vars()), train_vars)

        opt = optimizer(train_vars)

        start = timer()

        epoch_arr = []

        for i in range(epochs):
            grad, val = grad_fn()

            if callback is not None:
                callback(i, grad, val)

            opt(learning_rate, grad)

            epoch_arr.append(val)

        end = timer()
        training_time = end - start

        return epoch_arr, training_time
