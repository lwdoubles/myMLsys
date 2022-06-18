#ifndef MLSYS_BASE_H
#define MLSYS_BASE_H

#ifdef __cplusplus
#define MLSYS_BASE_EXTERN_C extern "C"
#else
#define MLSYS_BASE_EXTERN_C 
#endif

MLSYS_BASE_EXTERN_C{
    enum DeviceType{
        kCPU = 1,
        kGPU = 1
    };
    
    typedef struct {
        int device_id;
        DeviceType device_type;
    }MLContext;
    //张量定义
    typedef struct {
        void *data;
        MLContext ctx;
        int ndim;
        int64_t *shape;
    }Tensor;
}

#endif