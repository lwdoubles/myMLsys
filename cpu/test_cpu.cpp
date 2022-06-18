#include <cstdio>
#include <iostream>
#include "../mlsys_runtime.h"
#include "../mlsys_base.h"

using namespace std;

int main() {

    MLContext ctx;
    ctx.device_id = 0;
    ctx.device_type = kCPU;

    int ndim = 2;

    int64_t shape[]={4,2};

    auto *out = new TensorHandle();

    TensorMalloc(shape,ndim,ctx, out);

    cout<<(*out)->shape[0]<<" "<<(*out)->shape[1]<<endl;
    
    return 0;
}