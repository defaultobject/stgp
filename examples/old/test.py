import numpy as np
import scipy as sp
import legogp 
from legogp.computation.permutations import data_order_to_output_order
import matplotlib.pyplot as plt

Q = 2
N = 2
Nt = 3
A = sp.linalg.block_diag(*[(i+1) * np.ones([N, N]) for i in range(Q*Nt)])

P = data_order_to_output_order(Q, N*Nt)

breakpoint()
