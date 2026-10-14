import sympy as sp
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

# Это задание выполнено при помощи нейросети

t, s = sp.symbols('t s', real=True)
a, b, c, N0 = sp.symbols('a b c N0', positive=True)
C = sp.symbols('C')

N = sp.Function('N')
z = sp.Function('z')

# Исходное уравнение Бернулли
ode_N = sp.Eq(
    sp.diff(N(t), t),
    (a - b*t)*N(t) - c*N(t)**2
)

# После замены z = 1/N получаем линейное уравнение
ode_z = sp.Eq(
    sp.diff(z(t), t) + (a - b*t)*z(t),
    c
)
sol_z = sp.dsolve(ode_z, ics={z(0): 1/N0})
N_expr = sp.simplify(1 / sol_z.rhs)
print("\nРешение для N(t):")
sp.pprint(N_expr)


# -----------------------------
# 2. Функция для численного графика
# -----------------------------
def make_numeric_function(a_val, b_val, c_val, N0_val):
    expr_num = sp.simplify(
        N_expr.subs({
            a: a_val,
            b: b_val,
            c: c_val,
            N0: N0_val
        })
    )
    return sp.lambdify(t, expr_num, modules=['numpy', 'scipy'])

def find_peak_time(t_vals, y_vals):
    idx = np.nanargmax(y_vals)
    return t_vals[idx], y_vals[idx]

# -----------------------------
# 3. Базовые параметры
# -----------------------------
a_val = 1.0
N0_val = 1.0

# -----------------------------
# 4. Исследование b -> 0
# -----------------------------
c_val_fixed = 0.08
b_values = [0.3, 0.15, 0.08, 0.03, 0.01]

plt.figure(figsize=(10, 6))

for b_val in b_values:
    fN = make_numeric_function(a_val, b_val, c_val_fixed, N0_val)

    # Чем меньше b, тем дальше t*=a/b, поэтому окно берём адаптивно
    t_star = a_val / b_val
    t_max = min(2.0 * t_star, 80)   # ограничим сверху для удобства
    t_vals = np.linspace(0, t_max, 2000)

    N_vals = fN(t_vals)
    N_peak_t, N_peak_val = find_peak_time(t_vals, N_vals)

    plt.plot(t_vals, N_vals, label=f"b={b_val}, peak≈{N_peak_t:.2f}, t*=a/b={t_star:.2f}")

plt.title("Графики N(t) при малых b")
plt.xlabel("t")
plt.ylabel("N(t)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 5. Исследование c -> 0
# -----------------------------
b_val_fixed = 0.12
c_values = [0.3, 0.15, 0.08, 0.03, 0.01]

plt.figure(figsize=(10, 6))

for c_val in c_values:
    fN = make_numeric_function(a_val, b_val_fixed, c_val, N0_val)

    t_star = a_val / b_val_fixed
    t_max = 2.0 * t_star
    t_vals = np.linspace(0, t_max, 2000)

    N_vals = fN(t_vals)
    N_peak_t, N_peak_val = find_peak_time(t_vals, N_vals)

    plt.plot(t_vals, N_vals, label=f"c={c_val}, peak≈{N_peak_t:.2f}, t*=a/b={t_star:.2f}")

plt.axvline(a_val / b_val_fixed, color='black', linestyle='--', label='t*=a/b')
plt.title("Графики N(t) при малых c")
plt.xlabel("t")
plt.ylabel("N(t)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Численное сравнение времени пика и t*=a/b
# -----------------------------
print("\nИсследование b -> 0:")
for b_val in b_values:
    fN = make_numeric_function(a_val, b_val, c_val_fixed, N0_val)
    t_star = a_val / b_val
    t_max = min(2.0 * t_star, 80)
    t_vals = np.linspace(0, t_max, 3000)
    N_vals = fN(t_vals)
    peak_t, peak_val = find_peak_time(t_vals, N_vals)
    print(f"b={b_val:>5}: peak_t={peak_t:.4f}, t*=a/b={t_star:.4f}, difference={peak_t - t_star:.4f}")

print("\nИсследование c -> 0:")
for c_val in c_values:
    fN = make_numeric_function(a_val, b_val_fixed, c_val, N0_val)
    t_star = a_val / b_val_fixed
    t_vals = np.linspace(0, 2.0 * t_star, 3000)
    N_vals = fN(t_vals)
    peak_t, peak_val = find_peak_time(t_vals, N_vals)
    print(f"c={c_val:>5}: peak_t={peak_t:.4f}, t*=a/b={t_star:.4f}, difference={peak_t - t_star:.4f}")

