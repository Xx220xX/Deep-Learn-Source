# https://medium.com/binaryandmore/beginners-guide-to-deriving-and-implementing-backpropagation-e3c1a5a1e536
from gabriela_gpu.dnnCWrapper import *


class DNN:  # deep neural network
    def __init__(self, arch, hitLearn=0.21):
        self.gab = c_Gab()
        self.vetInp, self.vetOut = c.c_double * arch[0], c.c_double * arch[-1]
        n = c.c_int * len(arch)
        n, ln = n(*arch), len(arch)
        self.out = []
        self.sizeOut = arch[-1]
        clib.create_DNN(c.addressof(self.gab), n, ln, hitLearn)

    def __call__(self, input):
        temp = self.vetInp(*input)
        clib.call(c.addressof(self.gab), temp)
        out = self.vetOut()
        clib.getoutput(c.addressof(self.gab), out)
        self.out = [out[i] for i in range(self.sizeOut)]
        return self.out

    def aprender(self, true_output):
        temp = self.vetOut(*true_output)
        clib.learn(c.addressof(self.gab), temp)

    def learn(self, true_output):
        temp = self.vetOut(*true_output)
        clib.learn(c.addressof(self.gab), temp)

    def __del__(self):
        print('release gab')
        clib.release(c.addressof(self.gab))

    def save(self, file2save):
        raise ModuleNotFoundError("function not find")

    @staticmethod
    def load(file2load):
        raise ModuleNotFoundError("function not find")

    def randomize(self):
        clib.randomize(c.addressof(self.gab))
