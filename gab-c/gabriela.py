import os, ctypes as c
import time
from platform import architecture, system

__LOCAL_WORKS = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
__SYSTEM_ARCHITECTURE = '32'
__SYSTEM_EXTENSION_FORMAT = 'dll'
__SYSTEM_NAME = system()
if architecture()[0] == '64bit':
    __SYSTEM_ARCHITECTURE = '64'
if __SYSTEM_NAME == 'Darwin':
    raise OSError("module not compatible with your operating system")
elif __SYSTEM_NAME == "Linux":
    __SYSTEM_EXTENSION_FORMAT = 'so'

wrapper_c = c.CDLL(f'{__LOCAL_WORKS}/libgab{__SYSTEM_ARCHITECTURE}.{__SYSTEM_EXTENSION_FORMAT}')
wrapper_c.newDnn.argtypes = [c.c_void_p, c.c_int, c.c_double]
wrapper_c.newDnn.restype = c.c_void_p
wrapper_c.releaseDNN.argtypes = [c.c_void_p]
wrapper_c.call.argtypes = [c.c_void_p, c.c_void_p]
wrapper_c.learn.argtypes = [c.c_void_p, c.c_void_p]
wrapper_c.getOut.argtypes = [c.c_void_p, c.c_void_p]
wrapper_c.setSeed.argtypes = [c.c_int64]
wrapper_c.setSeed(time.time())

def setSeed(seed):
    wrapper_c.setSeed(seed)


class DNN:
    def __init__(self, architecture: tuple, hit_learn=0.1):
        size_arch = len(architecture)
        if size_arch<2 : raise IndexError('A deep neural network must have an input and an output, so the size of the architecture must be greater than or equal to 2')
        self.vector_input = c.c_double * architecture[0]
        self.vector_output = c.c_double * architecture[-1]
        cvector = c.c_int * size_arch
        architecture = cvector(*architecture)
        self.p = wrapper_c.newDnn(architecture, size_arch, hit_learn)
        wrapper_c.randomize(self.p)

    def __call__(self, input):
        wrapper_c.call(self.p, self.vector_input(*input))

    def learn(self, target):
        wrapper_c.call(self.p, self.vector_output(*target))

    @property
    def out(self) -> list:
        o = self.vector_output()
        wrapper_c.getOut(self.p, o)
        return list(o)

    def __del__(self):
        wrapper_c.releaseDNN(self.p)
if __name__ == '__main__':
    d = DNN((2,3,4,1))
    d([1,1])
    print(d.out)
