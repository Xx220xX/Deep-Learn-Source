from gabriela_gpu.dnn_gpu import DNN
from gabriela_gpu.dnn_gpu import setSeed
# for retro compatible version
RND = DNN

if __name__ == '__main__':  # teste

    input = [[1, 1], [1, 0], [0, 1], [0, 0]]
    target = [[int(i ^ j)] for i, j in input]
    print('creating ...')
    dnn = DNN((2,33,13,3, 1))
    print("create")
    epocas = 100
    error = 1e-3
    for ep in range(epocas):
        e = 0
        for i in range(len(input)):
            dnn(input[i])
            dnn.learn(target[i])
            e += (dnn.out[0] - target[i][0]) ** 2
        e /= 2

    print('after ', ep, 'epics i learn it, erro =', e)
    for i in range(len(input)):
        x, y = input[i]
        print(x, 'xor', y, '=', dnn(input[i]))



