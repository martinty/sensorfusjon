# %% imports
from typing import Tuple, Sequence, Any
from dataclasses import dataclass, field
from gradedINS.cat_slice import CatSlice

import numpy as np
import scipy.linalg as la

from gradedINS.quaternion import (
    euler_to_quaternion,
    quaternion_product,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
)

# from state import NominalIndex, ErrorIndex
from gradedINS.utils import cross_product_matrix

# %% Exam 2020



# %%
vv = np.array([3, 5])
vv1 = la.norm(vv)
vv_T = vv.reshape((2, 1))
vv2 = la.norm(vv_T)
vv3 = vv1 ** 2

# %%
R = np.array([
    [3, 4],
    [5, 1]
])

P = np.block([R, R])

# %%
G = np.array([
    [0, 0, 0, 0],
    [1, -1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
])


# %%
q = np.array([0, -1, 0, 0])
[roll, pitch, yaw] = quaternion_to_euler(q)

q2 = euler_to_quaternion(np.array([0, np.pi, np.pi]))


# %%
A = np.array([
    [0, -1, 0],
    [0,  0, 0],
    [0,  0, 1]
])
t = 2
B = la.expm(A*t)


# %%
A2 = np.array([
    [0, 0],
    [0, 1]
])
t2 = 2
B2 = la.expm(A2*t2)


# %%
q3 = np.array([0, 1, 0, 0])
w = np.array([0, np.pi, 0])
q_rhs = np.block([0, w])
q4 = 0.5 * quaternion_product(q3, q_rhs)


# %%
"""
v = np.array(
    [1, 2, 3]
)

# print(v[0])
# print(v[1])
# print(v[2])
# print(v)

m = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# print(m)
# print(m[0:2, 0:2])

A = v * m
B = v @ m
C = m @ v.T

v_T = v.T

D = np.array([
    v,
    v*2,
    v*3
])

E = v @ D

e = v @ v.T
f = v.T @ v
k = v[1:]

n = np.array([
    [1, 2, 3]
])

F = n.T @ n

m[1:3, 0:2] = np.zeros(2)


a = np.linalg.norm(n)
b = la.norm(n)

# %%
G = np.array([
    [1, 0, 0],
    [0, 5, 0],
    [0, 0, 9]
])
G_inv_np = np.linalg.inv(G)
G_inv_la = la.inv(G)

var = np.diag(np.array([3, 3, 3])**2)
"""
