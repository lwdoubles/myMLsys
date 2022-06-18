#include "cuda_base.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>



__global__ void set_value_kernel(float* tensor, float value, int kdim){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < kdim){
        tensor[index] = value;
    }
}

int set_value(TensorHandle tensor, float value){
    int kdim = 1;
    for(int i = 0; i < tensor->ndim; i++){
        kdim = kdim * tensor->shape[i];
    }

    float* tensor_data = (float *)tensor->data;

    int thread_per_block = 1024;
    int num_blocks = (kdim + thread_per_block - 1) / thread_per_block;

    set_value_kernel<< <num_blocks, thread_per_block> >>(tensor_data, value, kdim);
    return 0;
}


__global__ void broadcast_kernel(const float* input_data, float* output_data, index_t input_ndim, index_t output_ndim){
    index_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < output_ndim){
        output_data[idx] = input_data[idx % input_ndim];
    }
}


int broadcast(const TensorHandle input, TensorHandle output){
    index_t input_ndim = 1;
    for(int i = 0; i < input->ndim; i++){
        input_ndim *= input->shape[i];
    }

    index_t output_ndim = 1;
    for(int i = 0; i < output->ndim; i++){
        output_ndim *= output->shape[i];
    }

    const float* input_data = (const float*)input->data;
    float* output_data = (float*)output->data;

    int thread_per_block = 512;
    int num_blocks = (output_ndim + thread_per_block - 1) / thread_per_block;

    broadcast_kernel<< <num_blocks, thread_per_block> >>(input_data, output_data, input_ndim, output_ndim);

    return 0;
}


__global__ void reduced_sum_axis_zero_kernel(const float* input_data, float* output_data, int input_ndim, int output_ndim){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < output_ndim){
        output_data[idx] = 0;
        for(int i = 0; i < input_ndim / output_ndim; i++){
            output_data[idx] += input_data[i * output_ndim + idx];
        }
    }
}

int reduced_sum_axis_zero(const TensorHandle input, TensorHandle output){
    int input_ndim = 1;
    for(int i = 0; i < input->ndim; i++){
        input_ndim *= input->shape[i];
    }

    int output_ndim = 1;
    for(int i = 0; i < output->ndim; i++){
        output_ndim *= output->shape[i];
    }

    const float* input_data = (const float*)input->data;
    float* output_data = (float*)output->data;

    int thread_per_block = 1024;
    int num_blocks = (output_ndim + thread_per_block - 1) / thread_per_block;

    reduced_sum_axis_zero_kernel<< <num_blocks, thread_per_block> >>(input_data, output_data, input_ndim, output_ndim);

    return 0;
}


__global__ void matrix_elementwise_add_kernel(cosnt float*a, const float* b, float*c, int ndim){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < ndim){
        c[index] = a[index] + b[index];
    }
}

int matrix_elementwise_add(const TensorHandle a, const TensorHandle b, TensorHandle output){
    int ndim = 1;
    for(int i = 0; i < output->ndim; i++){
        n = n * output->shape[i];
    }
    const float* data_a = (const float*)a->data;
    const float* data_b = (const float*)b->data;
    float* data_output = (float*)output->data;

    int thread_per_block = 1024;
    int num_blocks = (ndim + thread_per_block - 1) / thread_per_block;

    matrix_elementwise_add<< <num_blocks, thread_per_block> >>(data_a, data_b, data_output, ndim);

    return 0;
}


__global__ void matrix_elementwise_subtract_kernel(cosnt float*a, const float* b, float*c, int ndim){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < ndim){
        c[index] = a[index] - b[index];
    }
}

int matrix_elementwise_substract(const TensorHandle a, const TensorHandle b, TensorHandle output){
    int ndim = 1;
    for(int i = 0; i < output->ndim; i++){
        n = n * output->shape[i];
    }
    const float* data_a = (const float*)a->data;
    const float* data_b = (const float*)b->data;
    float* data_output = (float*)output->data;

    int thread_per_block = 1024;
    int num_blocks = (ndim + thread_per_block - 1) / thread_per_block;

    matrix_elementwise_subtract_kernel<< <num_blocks, thread_per_block> >>(data_a, data_b, data_output, ndim);

    return 0;
}

__global__ void matrix_elementwise_mul_kernel(const float* a, const float* b, float* output, int ndim) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < ndim) {
        output[index] = a[index] * b[index];
    }
}

int matrix_elementwise_mul(const TensorHandle a, const TensorHandle b, TensorHandle output){
    int ndim = 1;
    for(int i = 0; i < output->ndim; i++){
        n = n * output->shape[i];
    }
    const float* data_a = (const float*)a->data;
    const float* data_b = (const float*)b->data;
    float* data_output = (float*)output->data;

    int thread_per_block = 1024;
    int num_blocks = (ndim + thread_per_block - 1) / thread_per_block;

    matrix_elementwise_mul_kernel<< <num_blocks, thread_per_block> >>(data_a, data_b, data_output, ndim);
}


__global__ void matrix_elementwise_division_kernel(const float* a, const float* b, float* output, int ndim){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < ndim){
        output[index] = a[index] / b[index];
    }
}

int matrix_elementwise_division(const TensorHandle a, const TensorHandle b, TensorHandle output){
    int ndim = 1;
    for(int i = 0; i < output->ndim; i++){
        ndim = ndim * output->shape[i];
    }
    const float* data_a = (const float*)a->data;
    const float* data_b = (const float*)b->data;
    float* data_output = (float*)output->data;

    int thread_per_block = 1024;
    int num_blocks = (ndim + thread_per_block - 1) / thread_per_block;

    matrix_elementwise_division_kernel<< <num_blocks, thread_per_block> >>(data_a, data_b, data_output, ndim);

    return 0;
}


__global__ void matrix_add_by_val_kernel(const float* mat, float* output, float val, int ndim){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < ndim){
        output[index] = mat[index] + val;
    }
}

int matrix_add_by_val(const TensorHandle a, float val, TensorHandle output){
    int ndim = 1;
    for(int i = 0; i < output->ndim; i++){
        ndim = ndim * output->shape[i];
    }
    const float* data_a = (const float*)a->data;
    const float* data_b = (const float*)b->data;
    float* data_output = (float*)output->data;

    int thread_per_block = 1024;
    int num_blocks = (ndim + thread_per_block - 1) / thread_per_block;

    matrix_add_by_val_kernel<< <num_blocks, thread_per_block> >>(data_a, data_b, data_output, ndim);

    return 0;
}



__global__ void matrix_substract_by_val_kernel(const float* mat, float* output, float val, int ndim){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < ndim){
        output[index] = mat[index] - val;
    }
}

int matrix_substract_by_val(const TensorHandle a, float val, TensorHandle output){
    int ndim = 1;
    for(int i = 0; i < output->ndim; i++){
        ndim = ndim * output->shape[i];
    }
    const float* data_a = (const float*)a->data;
    const float* data_b = (const float*)b->data;
    float* data_output = (float*)output->data;

    int thread_per_block = 1024;
    int num_blocks = (ndim + thread_per_block - 1) / thread_per_block;

    matrix_substract_by_val_kernel<< <num_blocks, thread_per_block> >>(data_a, data_b, data_output, ndim);

    return 0;
}

__global__ void matrix_mul_by_val_kernel(const float* mat, float* output, float val, int ndim){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < ndim){
        output[index] = output[index] * val;
    }
}

int matrix_mul_by_val(const TensorHandle input, float val, TensorHandle output){
    int ndim = 1;
    for(int i = 0; i < input->ndim; i++){
        ndim *= input->shape[i];
    }

    const float* input_data = (const float*)input->data;
    float* output_data = (float *)output->data;
    int thread_per_block = 1024;
    int num_blocks = (ndim + thread_per_block - 1) / thread_per_block;
    matrix_mul_by_val_kernel<< <num_blocks, thread_per_block> >>(input_data, output_data, val, ndim);

    return 0;
}


__global__ void matrix_div_by_val_kernel(const float* mat, float* output, float val, int ndim){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < ndim){
        output[index] = output[index] / val;
    }
}


int matrix_div_by_val(const TensorHandle input, float val, TensorHandle output){
    int ndim = 1;
    for(int i = 0; i < input->ndim; i++){
        ndim *= input->shape[i];
    }

    const float* input_data = (const float*)input->data;
    float* output_data = (float *)output->data;
    int thread_per_block = 1024;
    int num_blocks = (ndim + thread_per_block - 1) / thread_per_block;
    matrix_div_by_val_kernel<< <num_blocks, thread_per_block> >>(input_data, output_data, val, ndim);

    return 0;
}


/*
matA: m x k
matB: k x n
output: m x n
*/
int matrix_mul(const TensorHandle matA, const TensorHandle matB, TensorHandle output){
    int m = matA->shape[0], k = matA->shape[1], n = matB->shape[1];
    assert(matA->shape[1] == matB->shape[0]);

    int threadIdx = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

    if(threadIdx < m * n){
        int row = threadIdx / n;
        int col = threadIdx % n;

        output[threadIdx] = 0;

        for(int i = 0; i < k; i++){
            output[threadIdx] += matA[row * k + i] * matB[i * n + col];
        }
    }

    return 0;
}


__global__ void relu_kernel(const float *input, float *output, int n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n) {
        float element = input[index];
        if (element <= 0) {
            output[index] = 0;
        } else {
            output[index] = element;
        }
    }
}

int GpuRelu(const TensorHandle input, TensorHandle output) {
    int n = 1;
    for (int i = 0; i < input->ndim; i++) {
        n *= input->shape[i];
    }

    const float *input_data = (const float *) input->data;
    float *output_data = (float *) output->data;
    int threads_per_block = 1024;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    relu_kernel << < num_blocks, threads_per_block >> > (input_data, output_data, n);
    return 0;
}



__global__ void relu_gradient_kernel(const float* input, float* output, const float* grad, int n){
    int index = blockDim.x * blockIdx. + threadIdx.x;
    if(index < n){
        float element = input[index];
        if(element <= 0){
            output[index] = 0;
        }
        else{
            output[index] = grad[index];
        }
    }
}

int MLGpuReluGradient(const TensorHandle input, const TensorHandle grad, TensorHandle output){
    int n = 1;
    for(int i = 0; i < input->ndim; i++){
        n *= input->shape[i];
    }

    const float* input_data = input_data->data;
    float* output->data = output->data;
    const float* grad_data = grad->data;
    int thread_per_block = 1024;
    int num_blocks = (n + thread_per_block - 1) / thread_per_block;

    relu_gradient_kernel<< <num_blocks, thread_per_block>> >(input_data, output_data, grad_data, grad_data, n);

    return 0;
}


__global__ void softmax_kernel(int64_t nrow, int64_t ncol, const float* input_data, float* output_data){
    int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if(index >= nrow){
        return;
    }
    input_data += index * ncol;
    output_data += index * ncol;

    float max_num = *input_data;
    for(int x = 1; x < ncol; x++){
        max_num = max(max_num, input_data[x]);
    }
    float sum = 0;
    for(int x = 0; x < ncol; x++){
        sum += exp(input_data[x] - max_num);
    }
    for(int x = 0; x < ncol; x++){
        output_data[x] = exp(input_data[x] - max_num) / sum;
    }
}

int MLGpuSoftmax(const TensorHandle input, TensorHandle output){
    int64_t nrow = input->shape[0];
    int64_t ncol = input->shape[1];

    float* input_data = input->data;
    float* output_data = output->data;
    dim3 threads;

    if(nrow < 1024){
        threads.x = nrow;
    }
    else{
        threads.x = 1024;
        threads.y = (nrow + 1023) / 1024;
    }
    softmax_kernel<< <1, threads> >>(nrow, ncol, input_data, output_data);
    return 0;
}


// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  
  extern __shared__ float loss_per_row[];
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;

  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }

  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }

  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();

  float mean_loss = 0;

  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}


int MLGpuCrossEntropy(const TensorHandle input_a, const TensorHandle input_b, TensorHandle output) {
	int nrow = input_a->shape[0];
	assert(nrow <= 1024 * 4);
	int ncol = input_a->shape[1];
	const float *input_data_a = (const float *) input_a->data;
	const float *input_data_b = (const float *) input_b->data;
	float *output_data = (float *) output->data;
	dim3 threads;
	if (nrow <= 1024) {
		threads.x = nrow;
	} else {
		threads.x = 1024;
		threads.y = (nrow + 1023) / 1024;
	}
	matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
			nrow, ncol, input_data_a, input_data_b, output_data);
	return 0;
}