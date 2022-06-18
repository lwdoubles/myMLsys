import os
import ctypes

def load_lib():
    cur_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_path = os.path.join(cur_path, )
    path_to_so_file = os.path.join(lib_path, "xxx.so")

    lib = ctypes.CDLL(path_to_so_file, ctypes.RTLD_GLOBAL)

def c_array(ctype, values):
    return (ctype * len(values))(*values)

_LIB = load_lib()