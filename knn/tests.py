import numpy as np
from collections import Counter
from queue import PriorityQueue

x = np.array([[3,1],[2,8], [2,7], [5,2],[3,2],[8,2],[2,4]])
y = np.array([[1, -1, -1, 1, -1, 1, -1, 100, 4, 'sdaf']])
z = [-1]
unique, count = np.unique(y, return_counts=True)
counts = dict(zip(unique, count))
print(counts)
cnt = Counter(z)
print(cnt)
print(cnt.most_common()[0][1])
test = np.zeros((len(counts), len(counts)), dtype=int)
i = 0
class_number = {}
for key, value in counts.items():
    class_number[key] = i
    i += 1
print(class_number)

C = np.array([[1,3,5,3], [454,23,34,46], [324,234,234,54], [435,23,576,23]])
print(C)
print(C.sum())

"""
K = 2
q = PriorityQueue(K)
given_point = [1, 2]
data_points = [[1, 2], [200, 3], [-303, 49], [18, 18192]]
for index, point in enumerate(data_points):
    t = 0
    for i in range(len(given_point)):
        t += (point[i] - given_point[i]) ** 2

    if not q.full():
        q.put((-t, index))
    else:
        tmp = q.get()
        if -tmp[0] < t:
            q.put(tmp)
        else:
            q.put((-t, index))
res = []
while not q.empty():
    tp = q.get()
    res.append(tp[1])
print(res)
"""


