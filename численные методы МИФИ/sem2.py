import numpy as np

# метод вращения надо прочитать

m1 = np.random.randn(6,6)
print(m1, "\n")

m1_inv = np.linalg.inv(m1)

print(m1_inv, "\n")

m1_norm3 = max(m1.sum(axis=1))
m1_inv_norm3 = max(m1_inv.sum(axis=1))

# m1_eigvals, _ = np.linalg.eigh(m1)
# idx = np.argsort(m1_eigvals)[::-1]
# m1_eigvals = m1_eigvals[idx]
#
# m1_inv_eigvals, _ = np.linalg.eigh(m1_inv)
# idx_inv = np.argsort(m1_inv_eigvals)[:]
# m1_inv_eigvals = m1_inv_eigvals[idx_inv]

# print(f"собственые значения {m1_eigvals} и {m1_inv_eigvals}")

print(f"кубическая мера обусловленности: {m1_norm3*m1_inv_norm3}")
print(f'сферическая мера обусловленности: {np.linalg.cond(m1)}')

