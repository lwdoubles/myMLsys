import ctypes
from .load import _LIB
from . import ndarray

def array_set(tensor, value):
    _LIB.set_value(tensor.handle, ctypes.c_float(value))

def broadcast(input, output):
    _LIB.broadcast(input, output)

def reduced_sum_axis_zero(input, output):
    _LIB.reduced_sum_axis_zero(input.handle, output.handle)

def matrix_elementwise_add(mat_a, mat_b, mat_c):
    _LIB.matrix_elementwise_add(mat_a.handle, mat_b.handle, mat_c.handle)

def matrix_add_by_val(input, val, output):
    _LIB.matrix_add_by_val(input, val, output)

def matrix_elementwise_mul(mat_a, mat_b, mat_c):
    _LIB.matrix_elementwise_mul(mat_a, mat_b, mat_c)

def matrix_mul_by_val(input, val, output):
    _LIB.matrix_mul_by_val(input, val, output)

def matrix_mul(mat_a, mat_b, output):
    _LIB.matrix_mul(mat_a, mat_b, output)

def relu(input, output):
    _LIB.GpuRelu(input.handle, output.handle)

def relu_gradient(input, grad, output):
    _LIB.MLGpuReluGradient(input.handle, grad, output.handle)

def softmax(input, output):
    _LIB.MLGpuSoftmax(input.handle, output.handle)

def softmax_cross_entropy(input_a, input_b, output):
    _LIB.MLGpuCrossEntropy(input_a.handle, input_b.handle, output.handle)