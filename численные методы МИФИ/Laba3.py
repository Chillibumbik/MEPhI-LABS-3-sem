import numpy as np

def gauss_jordan_inverse(A, eps=1e-10):
    A = A.astype(float)
    n = A.shape[0]

    # Формируем расширенную матрицу [A | I]
    aug = np.hstack([A, np.eye(n)])
    det = 1.0
    swaps = 0

    # Прямой + обратный ход (Гаусс-Жордан)
    for k in range(n):
        # Выбор ведущего элемента
        max_row = np.argmax(np.abs(aug[k:, k])) + k
        if np.abs(aug[max_row, k]) < eps:
            raise ValueError("Матрица вырождена, обратная не существует.")
        if max_row != k:
            aug[[k, max_row]] = aug[[max_row, k]]
            det *= -1
            swaps += 1

        # Нормализация ведущей строки
        pivot = aug[k, k]
        det *= pivot
        aug[k, :] /= pivot

        # Обнуление остальных элементов в столбце
        for i in range(n):
            if i != k:
                factor = aug[i, k]
                aug[i, :] -= factor * aug[k, :]

    # После преобразований правая часть — обратная матрица
    A_inv = aug[:, n:]
    rang = np.sum(np.any(np.abs(aug[:, :n]) > eps, axis=1))
    return np.round(A_inv, 5), round(det, 5), rang, swaps, np.round(aug, 5)


def run_inverse_experiment(n=3, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Генерация случайной матрицы A ∈ [-0.5, 0.5]
    A = 0.5 - np.random.rand(n, n)
    print("=" * 70)
    print(f"Исходная матрица A (размерность {n}):")
    print(np.round(A, 4))

    # Находим обратную методом Гаусса-Жордана
    A_inv, det, rang, swaps, aug = gauss_jordan_inverse(A)
    print("\nОпределитель матрицы:", det)
    print("Ранг матрицы:", rang)
    print("Число перестановок строк:", swaps)
    print("\nОбратная матрица A^(-1):")
    print(np.round(A_inv, 4))

    # Проверка: A_inv * A ≈ I
    test = np.dot(A_inv, A)
    print("\nПроверка A^(-1) * A:")
    print(np.round(test, 4))

    # --- Решение системы через обратную матрицу ---
    b = 0.5 - np.random.rand(n)
    print("\nВектор правой части b:")
    print(np.round(b, 3))

    # Решение x = A_inv * b
    x = np.dot(A_inv, b)
    print("\nРешение x = A^(-1) * b:")
    print(np.round(x, 3))

    # Невязка dx = b - A * x
    dx = b - np.dot(A, x)
    print("\nНевязка dx = b - A*x:")
    print(np.round(dx, 5))
    print("Максимальная компонента невязки:", round(np.max(np.abs(dx)), 5))

    return {
        "A": np.round(A, 3),
        "A_inv": A_inv,
        "det": det,
        "rang": rang,
        "swaps": swaps,
        "test_identity": test,
        "b": np.round(b, 3),
        "x": np.round(x, 3),
        "residual": np.round(dx, 5)
    }


# ======== Примеры ========

# # n = 3 
# run_inverse_experiment(3, seed=1)


# n = 7 
results = run_inverse_experiment(7, seed=2)
print("\n=== РЕЗУЛЬТАТЫ ДЛЯ n=7 ===")
for k, v in results.items():
    print(f"{k}: {v}")
