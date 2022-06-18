#ifndef MLSYS_DEVICE_H
#define MLSYS_DEVICE_H

#include "mlsys_base.h"

#include<cassert>
#include<cstring>

class Device{
public:
    virtual ~Device() = default;

    virtual void* MallocData(MLContext ctx, size_t size, size_t alignment) = 0;

    virtual void FreeData(MLContext ctx, void* ptr) = 0;

    virtual void CopyData(const void* from, void* to, size_t size, MLContext ctx_from, MLContext ctx_to, MLStreamHandle stream) = 0;
};

#endif