def progress_bar_callback(num_epochs):
    """
        Simple progressbar - does not display learning objective value
    """
    from tqdm import tqdm


    bar = tqdm(total=num_epochs)

    def inner(epoch, grad, val):
        bar.update(1)

    return inner


def progress_bar_callback_notebook(num_epochs):
    """
        Simple progressbar - does not display learning objective value
    """
    from tqdm.notebook import trange, tqdm


    bar = tqdm(total=num_epochs)

    def inner(epoch, grad, val):
        bar.update(1)

    return inner

