from gabriela_gpu import DNN
from gabriela_gpu.dnn_gpu import setSeed

setSeed(1)
a = DNN((1, 3, 3, 1))
print(a([1]))
a.aprender([0])
print(a([1]))
setSeed(1)
a.randomize()
print(a([1]))
