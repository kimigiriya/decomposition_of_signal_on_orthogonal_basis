import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from math import factorial
from scipy import integrate
import warnings


warnings.filterwarnings('ignore', category=integrate.IntegrationWarning)

def lagger(f, n):
    t_sym = sp.symbols('t')
    f_sym = sp.sympify(f)

    L_n = []
    l_n = []
    p_t = sp.exp(-t_sym)
    for i in range(n):
        func_sym = t_sym**i * sp.exp(-t_sym)
        diff_i = sp.diff(func_sym, t_sym, i)
        L_n.append(sp.simplify(sp.exp(t_sym) / factorial(i) * diff_i))
        l_n.append(sp.simplify(sp.sqrt(p_t) * L_n[i]))
    print(f"Полином Лаггера: {L_n}")
    print(f"Ортонормированные функции Лаггера: {l_n}")

    c_n = []
    for i in range(n):
        integrand = f_sym * l_n[i]
        integrand_func = sp.lambdify(t_sym, integrand, 'numpy')
        integral, error = integrate.quad(lambda x: float(integrand_func(x)), 0, np.inf)

        c_n.append(integral)
    print(f"Коэффициенты: {c_n}")

    f_approx = sum(c_n[i] * l_n[i] for i in range(n))
    print(f"Ряд Фурье-Лаггера: {f_approx}")

    return f_approx.simplify(), f_sym

n = int(input("n = "))
b = int(input("b = "))  # Правая граница интервала [0, b]
f = input("f(t) = ")

if 0 > b:
    raise ValueError("Ошибка: b должно быть положительным!")

f_approx, f_sym = lagger(f, n)

t_dense = np.linspace(0.001, b, 100)
t_sym = sp.symbols('t')

f_num = sp.lambdify(t_sym, f_sym, 'numpy')
f_approx_num = sp.lambdify(t_sym, f_approx, 'numpy')

f_vals = f_num(t_dense)
f_approx_vals = f_approx_num(t_dense)

plt.figure(figsize=(8, 5))
plt.plot(t_dense, f_vals, 'm-', linewidth=2, label='Исходный')
plt.plot(t_dense, f_approx_vals, 'c--', linewidth=2, label='Разложение')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('Разложение по полиномам Лежандра')
plt.legend()
plt.grid(True)
plt.show()

#  -0.3*exp(-t) + exp(-0.3*t) + 0.3*exp(-0.3*t)
#  -0.4*exp(-1/3t) - exp(-5/8*t) + 3/2*exp(-t)
#  -2*t**4 - 6*t**3 + 7*t**2
#  cos(5*t) / (5*t)