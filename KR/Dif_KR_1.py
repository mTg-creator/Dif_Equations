import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

v = sp.Function('v')
t, k, m, F = sp.symbols('t k m F')
ode_v = sp.Eq(
    sp.diff(v(t), t) + k/m * v(t), -F/m)
res = sp.dsolve(ode_v, v(t), ics={v(0): 0})
sp.pprint(res)
