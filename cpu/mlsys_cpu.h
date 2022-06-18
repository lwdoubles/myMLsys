#ifndef MLSYS_CPU_H
#define MLSYS_CPU_H

#include "../mlsys_runtime.h"
#include "../mlsys_device.h"
#include <cassert>
#include <cstring>

class MLSYS_CPU: public Device{
public:
    void* MallocData(MLContext ctx, size_t size, size_t alignment);
    void FreeData(MLContext ctx, void* ptr);
    void CopyData(const void* from, void* to, size_t size, MLContext ctx_from, MLContext ctx_to, MLStreamHandle stream);
    void StreamSync(MLContext ctx, MLStreamHandle stream);
};

#endif




