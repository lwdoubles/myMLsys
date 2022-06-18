#ifndef CUDA_BASE_H
#define CUDA_BASE_H

#include "../mlsys_runtime.h"
#include "mlsys_device.h"
#include <cuda_runtime.h>
#include <cassert.h>
#include <string>

class CUDA_BASE: public Device {
public:
    void* CudaMalloc(MLContext ctx, size_t size, size_t alignment);

    void CudaFree(MLContext ctx, void* ptr);

    void CudaCopy(const void* from, void* to, size_t size, MLContext ctx_from, MLContext ctx_to, MLStreamHandle stream);

    void StreamSync(MLContext ctx, MLStreamHandle stream);
};