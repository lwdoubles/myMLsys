#include "mlsys_cpu.h"
#include <cstdlib>
#include <iostream>

void* MLSYS_CPU::MallocData(MLContext ctx, size_t size, size_t alignment){
    void* ptr;
    //调用posix_memalign( )成功时会返回size字节的动态内存，并且这块内存的地址是alignment的倍数。参数alignment必须是2的幂，还是void指针的大小的倍数。返回的内存块的地址放在了memptr里面，函数返回值是0.
    int ret = posix_memalign(&ptr, alignment, size);
    if(ret != 0)
        throw std::bad_alloc();
    return ptr;
}
 
 
void MLSYS_CPU::FreeData(MLContext ctx, void* ptr){
    free(ptr);
}

void MLSYS_CPU::CopyData(const void *from, void* to, size_t size, MLContext ctx_from, MLContext ctx_to, MLStreamHandle stream){
    memcpy(to, from, size);
}

void MLSYS_CPU::StreamSync(MLContext ctd, MLStreamHandle stream){
    
}