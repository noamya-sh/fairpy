import time
import numpy as np
import pyximport
from improve_performance_algo2.cython_algorithm2 import envy_free_approximation_cython
from fairpy import ValuationMatrix, AllocationMatrix, Allocation
from envy_free_approximation_division import envy_free_approximation
import matplotlib.pyplot as plt

pyximport.install()

cy = []
py = []
idx = 0
for i in range(5, 280, 5):
    st_l = time.time()
    shape = (i, i)
    v = np.random.randint(-i, i, size=shape)
    a = np.eye(i, k=0, dtype=int)
    v2 = np.copy(v)
    alloc = Allocation(agents=ValuationMatrix(v), bundles=AllocationMatrix(a))
    alloc2 = Allocation(agents=ValuationMatrix(v2), bundles=AllocationMatrix(a))
    st = time.time()
    envy_free_approximation_cython(alloc, 0.1)
    cy.append(time.time() - st)
    st = time.time()
    envy_free_approximation(alloc2, 0.1)
    py.append(time.time() - st)
    print(i, "time iter:", time.time() - st_l, "|time py:", py[idx], "|time cy:", cy[idx])
    idx += 1
x = [i for i in range(5, 280, 5)]
plt.plot(x, py, color='gold')
plt.plot(x, cy, color='navy')
plt.xlabel("No. agents and bundles")
plt.ylabel("Time (in sec)")
plt.legend(["python", "cython"])
plt.title("Comparison of running Python Vs Cython versions on random instances")
plt.figure(figsize=(14, 8))
plt.show()
