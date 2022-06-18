#ifndef MLSYS_RUNTIME_H
#define MLSYS_RUNTIME_H


#ifdef __cplusplus
#define MLSYS_EXTERN_C extern "C"
#else
#define MLSYS_BASE_EXTERN_C
#endif

#include<cstdint>
#include<cstddef>

#include "mlsys_base.h"

MLSYS_EXTERN_C{
    typedef int64_t index_t;
    typedef Tensor* TensorHandle;
    typedef void* MLStreamHandle;

    int TensorMalloc(const index_t *shape, int ndim, MLContext ctx, TensorHandle *out);
    int TensorFree(TensorHandle handle);
    int MLcopy(TensorHandle from, TensorHandle to, MLStreamHandle stream);
}
#endif