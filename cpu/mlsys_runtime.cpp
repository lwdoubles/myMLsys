//#include "../mlsys_runtime.h"
#include "mlsys_cpu.h"
#include <array>
#include <thread>
#include <iostream>


class DeviceManager{
public:
    static constexpr int kDeviceMaxNum = 8;

    static Device* Get(MLContext ctx){
        return Global()->getDevice(ctx.device_type);
    }

private:
    std::array<Device*, kDeviceMaxNum> device;

    DeviceManager(){
        std::fill(device.begin(), device.end(), nullptr);
        static MLSYS_CPU cpu_device;
        device[kCPU] = static_cast<Device*>(&cpu_device);
    }

    static DeviceManager* Global(){
        static DeviceManager device_manager;
        return &device_manager;
    }

    Device* getDevice(DeviceType type){
        if(device[type] == nullptr){
            exit(EXIT_FAILURE);
        }
        return device[type];
    }


};

Tensor* newTensor(){
    auto* arr = new Tensor();
    arr->shape = nullptr;
    arr->ndim = 0;
    arr->data = nullptr;
    return arr;
}

void TensorFree_(Tensor* arr){
    if(arr != nullptr){
        delete []arr->shape;
        if(arr->data != nullptr){
            DeviceManager::Get(arr->ctx)->FreeData(arr->ctx, arr->data);
        }
    }
    delete arr;
}

size_t tensor_size(Tensor* tensor){
    size_t size = 1;
    for(index_t i = 0; i < tensor->ndim; i++){
        size *= tensor->shape[i];
    }
    size *= 4;
    //std::cout<<size<<endl;
    return size;
}

size_t tensor_alignment(Tensor *tensor) {
    return 8;
}

int TensorMalloc(const index_t* shape, int ndim, MLContext ctx, TensorHandle* out){
    Tensor* tensor = nullptr;

    tensor = newTensor();
    tensor->ndim = ndim;
    auto* shape_copy = new index_t[ndim];
    std::copy(shape, shape+ndim, shape_copy);

    tensor->shape = shape_copy;
    tensor->ctx = ctx;
    size_t size = tensor_size(tensor);
    size_t alignment = tensor_alignment(tensor);

    tensor->data = DeviceManager::Get(ctx)->MallocData(ctx, size, alignment);
    *out = tensor;
    return 1;
}

int TensorFree(TensorHandle handle){
    Tensor* tensor = handle;
    TensorFree_(tensor);
}

int MLCopy(TensorHandle from, TensorHandle to, MLStreamHandle stream){
    size_t from_size = tensor_size(from);
    size_t to_size = tensor_size(to);

    assert(from_size == to_size);

    MLContext ctx = from->ctx;
    if(ctx.device_type == kCPU){
        ctx = to->ctx;
    }
    DeviceManager::Get(ctx)->CopyData(from->data, to->data, from_size, from->ctx, to->ctx, stream);
}
