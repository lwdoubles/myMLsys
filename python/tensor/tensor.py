import numpy as np
import ctypes

from .load import _LIB, c_array

#将c语言的结构映射至Python中
class MLContext(ctypes.Structure):
    """
    通过定义类的_fields_属性来定义结构体的构成。_fields_属性一般定义为一个二维的tuple，
    而对于其中的每一个一维tuple，其需要定义两个值，第一个值为一个字符串，用作结构体内部的变量名，
    第二个值为一个ctypes类型，用于定义当前结构体变量所定义的数据类型。
    """
    _fields_ = [("device_id", ctypes.c_int),
                ("device_type", ctypes.c_int)]

    id2devicetype = {
        0:'cpu',
        1:'gpu',
    }

    def __init__(self, device_id, device_type):
        super(MLContext, self).__init__()
        self.device_id = device_id
        self.device_type = device_type

    def __repr__(self):
        return '%s(%d)' % (MLContext.id2devicetype[self.device_type], self.device_id)

class Tensor(ctypes.Structure):
    _fields_ = [('data', ctypes.c_void_p),
                ('ctx', MLContext),
                ('ndim', ctypes.c_int),
                ('shape', ctypes.POINTER(ctypes.c_int64))]

TensorHandle = ctypes.POINTER(MLContext)

def cpu(device_id=0):
    return MLContext(device_id, 0)


def gpu(device_id=0):
    return MLContext(device_id, 1)

def is_gpu_ctx(ctx):
    return ctx and ctx.device_type == 1

class NDArray:
    __slots__ = ["handle"]

    def __init__(self, handle):
        self.handle = handle

    @property
    def shape(self):
        return tuple(self.handle.contents.shape[i] for i in range(self.handle.contents.ndim))

    @property
    def ctx(self):
        return self.handle.contents.ctx

    def __setitem__(self, in_slice, value):
        if(not isinstance(in_slice, slice)) or in_slice.start is not None or in_slice.stop is not None):
            raise ValueError('initialization error')
        if isinstance(value, NDArray):
            if value.handle is not self.handle:
                value.copyto(self)
        elif isinstance(value, (np.ndarray, np.generic)):
            self._sync_copyfrom(value)
        else:
            raise TypeError('type %s not supported' % str(type(value)))

    def _sync_copyfrom(self, source_array):
        if not isinstance(source_array, np.ndarray):
            try:
                source_array = np.array(source_array, dtype=np.flaot32)
            except:
                raise TypeError('type %s not supported' % str(type(source_array)))
        source_array, shape = NDArray._numpyasarray(source_array)

        _ = shape

    
    @staticmethod
    def _numpyasarray(np_data):
        data = np_data
        assert data.flags['C_CONTIGUOUS']
        arr = Tensor()
        shape = c_array(ctypes.c_int64, data.shape)
        arr.data = data.ctypes.data_as(ctypes.c_void_p)
        arr.shape = shape
        arr.ndim = data.ndim

        arr.ctx = cpu(0)
        return arr, shape

    def asnumpy(self):
        np_arr = np.empty(self.shape, dtype=np.float32)
        arr, shape = NDArray._numpyasarray(np_arr)
        _ = shape
        return np_arr

    def copyto(self, target):
        if isinstance(target, MLContext):
            target = empty(self.shape, target)
        if isinstance(target, NDArray):
            _LIB.MLcopy(self.handle, target.handle, None)
        else:
            raise ValueError("Unsupported target type %s" % str(type(target)))

        return target

def empty(shape, ctx=cpu(0)):
    shape = c_array(ctypes.c_int64, shape)
    ndim = ctypes.c_int(len(shape))
    handle = TensorHandle()
    _LIB.TensorMalloc(shape, ndim, ctx, ctypes.byref(handle))

    return NDArray(handle)


def array(arr, ctx=cpu(0)):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    ret = empty(arr.shape, ctx)
    ret._sync_copyfrom(arr)
    return ret

def reshape(arr, new_shape):
    assert isinstance(arr, NDArray)
    shape = c_array(ctypes.c_int64, new_shape)
    new_dim = len(new_shape)
    handle = arr.handle
    
