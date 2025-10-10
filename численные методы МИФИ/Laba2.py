import numpy as np

def gauss_solve(A, b, use_pivot=True, eps=1e-10):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    aug = np.hstack([A, b.reshape(-1, 1)])  # расширенная матрица
    det = 1.0
    swaps = 0

    # Прямой ход
    for k in range(n - 1):
        # Выбор ведущего элемента
        if use_pivot:
            max_row = np.argmax(np.abs(aug[k:, k])) + k
            if np.abs(aug[max_row, k]) < eps:
                det = 0
                break
            if max_row != k:
                aug[[k, max_row]] = aug[[max_row, k]]
                swaps += 1
                det *= -1  # смена знака определителя
        else:
            if np.abs(aug[k, k]) < eps:
                det = 0
                break

        det *= aug[k, k]
        for i in range(k + 1, n):
            factor = aug[i, k] / aug[k, k]
            aug[i, k:] -= factor * aug[k, k:]

    # Последний элемент для определителя
    if np.abs(aug[n - 1, n - 1]) < eps:
        det = 0
    else:
        det *= aug[n - 1, n - 1]

    # Ранг
    rang = np.sum(np.any(np.abs(aug[:, :-1]) > eps, axis=1))

    # Обратный ход
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if np.abs(aug[i, i]) < eps:
            x[i] = np.nan
            continue
        x[i] = (aug[i, -1] - np.dot(aug[i, i + 1:n], x[i + 1:n])) / aug[i, i]

    return x, det, rang, swaps, aug


def run_gauss_experiment(n=8, seed=None):
    if seed is not None:
        np.random.seed(seed)

    A = 0.5 - np.random.rand(n, n)
    b = A.sum(axis=1)

    print("=" * 70)
    print(f"Исходная матрица A (размерность {n}):")
    print(np.round(A, 3))
    print("\nВектор b:")
    print(np.round(b, 3))
    print("=" * 70)

    print(f"\n=== Решение системы размерности {n} с перестановкой строк ===")
    x1, det1, rang1, swaps1, aug1 = gauss_solve(A, b, use_pivot=True)
    print("Определитель:", round(det1, 3))
    print("Ранг:", rang1)
    print("Число перестановок строк:", swaps1)
    print("Решение:", np.round(x1, 3))

    print("\n=== Решение без перестановки строк ===")
    x2, det2, rang2, swaps2, aug2 = gauss_solve(A, b, use_pivot=False)
    print("Определитель:", round(det2, 3))
    print("Ранг:", rang2)
    print("Число перестановок строк:", swaps2)
    print("Решение:", np.round(x2, 3))

    dx = x1 - x2
    norm_cubic = np.max(np.abs(dx))
    print("\nРазность двух решений:", np.round(dx, 3))
    print("Кубическая норма разности решений:", round(norm_cubic, 3))

    return {
        "x_with_pivot": np.round(x1, 3),
        "x_without_pivot": np.round(x2, 3),
        "det_with_pivot": round(det1, 3),
        "det_without_pivot": round(det2, 3),
        "rang_with_pivot": rang1,
        "rang_without_pivot": rang2,
        "swaps_with_pivot": swaps1,
        "swaps_without_pivot": swaps2,
        "norm_diff": round(norm_cubic, 3)
    }


# ======== Запуск ========
# n=8 (демо)
# run_gauss_experiment(8, seed=1)

# n=16 (по заданию)
results = run_gauss_experiment(16, seed=2)
print("\n=== РЕЗУЛЬТАТЫ ДЛЯ n=16 ===")
for k, v in results.items():
    print(f"{k}: {v}")
