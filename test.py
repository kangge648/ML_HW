import numpy as np

test = np.zeros(5)
test1 = np.ones(5)
test1 = test1.reshape(-1, 1)
test2 = 2* np.ones(5)
print(test.shape)
print(test)
print(test1.shape)
print(test1)
# res = []
# res.append(test)
# res.append(test1)
# res.append(test2)
# res = np.array(res)
# print(res)