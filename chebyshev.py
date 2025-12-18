import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from math import factorial
from scipy import integrate
import warnings


warnings.filterwarnings('ignore', category=integrate.IntegrationWarning)

def chebyshev(f, n):
    t_sym = sp.symbols('t')
    f_sym = sp.sympify(f)

    T_n = []
    for i in range(n):
        func_sym = sp.sqrt(1 - t_sym ** 2) ** (2 * i - 1)
        diff_i = sp.diff(func_sym, t_sym, i)
        T_n.append(sp.simplify(((-2)**i) * factorial(i) / factorial(2*i) * sp.sqrt(1 - t_sym**2) * diff_i))
    print(f"Полином Чебышева: {T_n}")

    c_n = []
    integrand = f_sym / sp.sqrt(1 - t_sym**2)
    integrand_func = sp.lambdify(t_sym, integrand, 'numpy')
    integral, error = integrate.quad(lambda x: float(integrand_func(x)), -1, 1)
    c_n.append(1 / np.pi * integral)  # c_0

    for i in range(1, n):
        integrand = f_sym * T_n[i] / sp.sqrt(1 - t_sym**2)
        integrand_func = sp.lambdify(t_sym, integrand, 'numpy')
        integral, error = integrate.quad(lambda x: float(integrand_func(x)), -1, 1)

        c_n.append(2 / np.pi * integral)
    print(f"Коэффициенты: {c_n}")

    f_approx = sum(c_n[i] * T_n[i] for i in range(n))  # c_0 * T_0 = c_0
    print(f"Ряд Фурье-Чебышева: {f_approx}")

    return f_approx.simplify(), f_sym

n = int(input("n = "))
f = input("f(t) = ")

f_approx, f_sym = chebyshev(f, n)

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
plt.title('Разложение по полиномам Чебышева I рода')
plt.legend()
plt.grid(True)
plt.show()

#  3/4*t**3 - 5/4*t**2 + 1/4*t
#  5*t**5 - 2/3*t**2 + 1
#  2*t**4 - 3*t**3 + 2*t**2 - 3*t
#  50*t**4 - 10*t + 2