import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from math import factorial
from scipy import integrate
import warnings


warnings.filterwarnings('ignore', category=integrate.IntegrationWarning)

def hermite(f, n):
    t_sym = sp.symbols('t')
    f_sym = sp.sympify(f)

    H_n = []
    h_n = []
    for i in range(n):
        func_sym = sp.exp(-t_sym**2 / 2)
        diff_i = sp.diff(func_sym, t_sym, i)
        H_n.append(sp.simplify((-1)**i * sp.exp(t_sym**2 / 2) * diff_i))
        h_n.append((H_n[i] * sp.exp(-t_sym**2 / 2)) / sp.sqrt(2**i * factorial(i) * sp.sqrt(sp.pi)))
    print(f"Полином Эрмита: {H_n}")
    print(f"Ортонормированные функции Эрмита: {h_n}")

    c_n = []
    for i in range(n):
        integrand = f_sym * h_n[i]
        integrand_func = sp.lambdify(t_sym, integrand, 'numpy')
        integral, error = integrate.quad(lambda x: float(integrand_func(x)), -np.inf, np.inf)

        c_n.append(integral)
    print(f"Коэффициенты: {c_n}")

    f_approx = sum(c_n[i] * h_n[i] for i in range(n))
    print(f"Ряд Фурье-Эрмита: {f_approx}")

    return f_approx.simplify(), f_sym

n = int(input("n = "))
a = int(input("a = "))  # Левая граница интервала [a, b]
b = int(input("b = "))  # Правая граница интервала [a, b]
f = input("f(t) = ")

if a > b:
    raise ValueError("Ошибка: a должно быть меньше b!")

f_approx, f_sym = hermite(f, n)

t_dense = np.linspace(a, b, 100)
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
plt.title('Разложение по полиномам Эрмита')
plt.legend()
plt.grid(True)
plt.show()

#  exp(-t**2/2) / pi**(1/4)
#  (exp(-t**2/2) / pi**(1/4) + sqrt(2)*(t**2 - 1)*exp(-t**2/2)/(4*pi**(1/4)))
#  sqrt(2)*t*exp(-t**2/2)/pi**(1/4)
#  sqrt(2)*(t**2 - 1)*exp(-t**2/2)/(4*pi**(1/4))
#  exp(-t**2/2)/pi**(1/4) + 0.1*sqrt(2)*t*exp(-t**2/2)/pi**(1/4)
#  exp(-t**2/2)/pi**(1/4) + 0.05*sqrt(2)*(t**2 - 1)*exp(-t**2/2)/(4*pi**(1/4))