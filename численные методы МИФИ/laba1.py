import numpy as np
import matplotlib.pyplot as plt

def generate_spd_matrix(n):
    """Генерирует случайную симметричную положительно определенную матрицу."""
    M = np.random.rand(n, n) - 0.5
    A = np.dot(M.T, M)  # симметричная и положительно определённая
    return A

def sor(A, b, omega, eps=1e-6, max_iter=5000):
    """Метод верхней релаксации (SOR).
       Возвращает найденное решение, число итераций и код завершения."""
    n = len(b)
    x = np.zeros_like(b)
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (1 - omega) * x_old[i] + (omega / A[i, i]) * (b[i] - s1 - s2)

        if np.linalg.norm(x - x_old, ord=np.inf) < eps:
            return x, k + 1, 0  # успешно
    return x, max_iter, 1  # не сошлось

def experiment(n):
    A = generate_spd_matrix(n)
    print(f"Исходная матрица:\n{A}")
    # единичное решение x = [1,1,...,1]
    x_true = np.ones(n)
    b = A @ x_true

    omegas = np.linspace(0.05, 2.0, 40)
    iters = []
    valid_omegas = []

    for omega in omegas:
        _, k, code = sor(A, b, omega)
        if code == 0:  # решение сошлось
            iters.append(k)
            vaMlid_omegas.append(omega)

    # Поиск оптимального
    if iters:
        k_min = min(iters)
        t_opt = valid_omegas[iters.index(k_min)]
    else:
        k_min, t_opt = None, None

    # Построение графика
    plt.plot(valid_omegas, iters, marker='o')
    plt.xlabel("Параметр релаксации t (omega)")
    plt.ylabel("Число итераций")
    plt.title(f"Метод верхней релаксации (n={n})")
    plt.grid(True)
    plt.show()

    print(f"n={n}, минимальное число итераций = {k_min}, оптимальное t = {t_opt:.3f}")

# Запуск экспериментов
experiment(3)
experiment(6)
