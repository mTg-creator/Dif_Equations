import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
# Переменные и параметры
t = sp.symbols('t', nonnegative=True)
m, k, P0, tau, v0 = sp.symbols(
    'm k P0 tau v0',
    positive=True
)

v = sp.Function('v')
u = sp.Function('u')

# Исходное уравнение Бернулли
ode_v = sp.Eq(
    sp.diff(v(t), t),
    P0 * sp.exp(-t / tau) / (m * v(t)) - k * v(t) / m)

ode_u = sp.Eq(
    sp.diff(u(t), t) + 2*k/m*u(t),
    2*P0/m * sp.exp(-t/tau)
)

# Решение с условием u(0) = v0^2
sol_u = sp.dsolve(
    ode_u,
    ics={u(0): v0**2}
)
v_solution = sp.sqrt(sp.simplify(sol_u.rhs))
print("Ответ v(t):")
sp.pprint(v_solution)
# Численные значения параметров
params = {
    m: 1000.0,
    k: 80.0,
    P0: 30000.0,
    v0: 2.0
    }

tau_values = [1, 2, 5, 10, 20]
t_vals = np.linspace(0, 40, 1000)


plt.figure(figsize=(10, 6))
for tau_val in tau_values:
    # Подставляем параметры в решение SymPy
    v_tau_expr = sp.simplify(v_solution.subs(params).subs(tau, tau_val))

    # Превращаем в численную функцию
    v_tau_func = sp.lambdify(t, v_tau_expr, 'numpy')

    # Считаем значения
    v_vals = v_tau_func(t_vals)

    plt.plot(t_vals, v_vals, label=fr'$\tau={tau_val}$')

plt.xlabel('Время t, с', fontsize=12)
plt.ylabel('Скорость v(t), м/с', fontsize=12)
plt.title('Зависимость скорости катера от времени', fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
