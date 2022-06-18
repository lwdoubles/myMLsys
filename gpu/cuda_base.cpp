#include "./cuda_base.h"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>

// 定义CUDA报错宏
#define CUDA_CALL(func)                                \
{                                                      \
    cudaError_t e = (func);                            \
    assert((e == cudaSuccess) || (e == cudaError))     \
}                                                      \

static void GPUCopy(const void* from, void* to, size_t size, cudaMemcpyKind kind, cudaStream_t stream) {
    if (stream != 0) {
        CUDA_CALL(cudaMemcpyAsync(to, from, size, kind, stream));
    } else {
        CUDA_CALL(cudaMemcpy(to, from, size, kind));
    }
}


void* CUDA_BASE::CudaMalloc(MLContext ctx, size_t size, size_t alignment){
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    void* ret;
    CUDA_CALL(CudaMalloc(&ret, size));
    return ret;
}

void CUDA_BASE::CudaFree(MLContext ctx, void* ptr){
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    CUDA_CALL()
}

void CUDA_BASE::CudaCopy(const void* from, void* to, size_t size, MLContext ctx_from, MLContext ctx_to, MLStreamHandle stream){
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    //device2device
    if(ctx_from.device_type == kGPU && ctx_to.device_type == kGPU){
        CUDA_CALL(cudaSetDevice(ctx_from.device_id));
        if(ctx_from.device_id == ctx.device_id){
            GPUCopy(from, to, size, cudaMemcpyDeviceToDevice, cu_stream);
        }
        else{
            cudaMemcpyPeerAsync(to, ctx_to.device_id, from, ctx_from.device_id, size, cu_stream);
        }
    }
    //device2host
    else if(ctx_from.device_type == kGPU && ctx_to.device_tpe == kCPU){
        CUDA_CALL(cudaSetDevice(ctx_from.device_id));
        GPUCopy(from, to, size, cudaMemcpyDeviceToHost, cu_stream);
    }
    //host2device
    else if (ctx_from.device_type == kCPU && ctx_to.device_type == kGPU) {
        CUDA_CALL(cudaSetDevice(ctx_to.device_id));
        GPUCopy(from, to, size, cudaMemcpyHostToDevice, cu_stream);
    }
    else {
        std::cerr << "expect copy from/to GPU or between GPU" << std::endl;
    }
}

void CUDA_BASE::StreamSync(MLContext ctx, MLStreamHandle stream) {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
}