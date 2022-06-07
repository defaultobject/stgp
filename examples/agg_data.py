import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def aggregate_in_time(x, y, group_size):
    """
    Groups (x, y) in groups of group_size. Y is returned as the average of the group.
    """

    N = x.shape[0]

    aggr_x = []
    aggr_y = []

    current_index = 0
    for i in range(int(N/group_size)):
        next_index = current_index + group_size
        aggr_x.append(x[current_index:next_index])

        aggr_y.append(
            np.sum(y[current_index:next_index])/(group_size)
        )

        current_index = next_index

    return np.array(aggr_x), np.array(aggr_y)

def plot_timeseries_aggregated_xy(x, y):
    linecolor= 'blue'

    for n in range(y.shape[0]):
        # get group boundary
        min_x, max_x = np.min(x[n]), np.max(x[n])
        plt.plot([min_x, max_x], [y[n], y[n]], c=linecolor)


x = np.linspace(0, 1, 100)
f = np.sin(x*10) 

x_aggr, f_aggr = aggregate_in_time(x, f, 5)
y_aggr = f_aggr + 0.01*np.random.randn(f_aggr.shape[0])

if True:
    plt.plot(x, f)
    plot_timeseries_aggregated_xy(x_aggr, y_aggr)
    plt.show()
