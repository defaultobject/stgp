def progress_bar_callback(num_epochs):
    """
        Simple progressbar - does not display learning objective value
    """
    from tqdm import tqdm


    bar = tqdm(total=num_epochs)

    def inner(epoch, grad, val):
        bar.update(1)

    return inner

