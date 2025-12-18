import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from math import factorial
from scipy import integrate
import warnings


warnings.filterwarnings('ignore', category=integrate.IntegrationWarning)

def legendre(f, n):
    t_sym = sp.symbols('t')

    P_n = []
    for i in range(n):
        func_sym = (t_sym**2 - 1)**i
        diff_i = sp.diff(func_sym, t_sym, i)
        P_n.append(sp.simplify((1 / (2**i * factorial(i))) * diff_i))
    print(f"Полином Лежандра: {P_n}")

    c_n = []
    f_sym = sp.sympify(f)
    for i in range(n):
        integrand = f_sym * P_n[i]
        integrand_func = sp.lambdify(t_sym, integrand, 'numpy')
        integral, error = integrate.quad(lambda x: float(integrand_func(x)), -1, 1)

        c_n.append((2 * i + 1) / 2 * integral)
    print(f"Коэффициенты: {c_n}")

    f_approx = sum(c_n[i] * P_n[i] for i in range(n))
    print(f"Ряд Фурье-Лежандра: {f_approx}")

    return f_approx.simplify(), f_sym

n = int(input("n = "))
f = input("f(t) = ")

f_approx, f_sym = legendre(f, n)

t_dense = np.linspace(-1, 1, 100)
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

#  0.5*exp(0.2*t) - 0.2*exp(0.5*t)
#  exp(-t/2) + exp(t)
#  -exp(0.7*t) + exp(0.3*t)
#  1 / (50*t**2 - 10*t + 2)