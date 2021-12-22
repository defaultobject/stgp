import numpy as np

sig_y = 1.4

K = np.ones([4, 4]) + np.eye(4) * 0.1

L = np.eye(3) * sig_y + np.eye(3)*0.1
L_tilde = np.eye(4) * sig_y + np.eye(4)*0.1

L = np.ones([3, 3]) * sig_y
L_tilde =np.ones([4, 4]) * sig_y

A = np.eye(3)
A = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
])

A_tilde = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
])

print(A.T @ L @ A)

print(A_tilde.T @ L_tilde @ A_tilde)

a = np.array([1, 1, 0, 1])[:, None]

print(a @ a.T)

target = np.linalg.inv(np.linalg.inv(K) + np.linalg.inv(A.T @ L @ A))
target_tilde = np.linalg.inv(np.linalg.inv(K) + np.linalg.inv(A_tilde.T @ L_tilde @ A_tilde))
print(target)
print(target_tilde)

breakpoint()
