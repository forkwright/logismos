// Row-wise softmax. One workgroup per row. Two-pass reduction in
// LDS: max, then sum of exp(x - max).

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

extern "C" __global__ void logismos_softmax_fp16_kernel(
    const __half* __restrict__ x,
    __half*       __restrict__ y,
    int m,
    int n)
{
    int row = blockIdx.x;
    if (row >= m) return;

    const __half* x_row = x + row * n;
    __half*       y_row = y + row * n;

    int tid = threadIdx.x;
    int bs  = blockDim.x;

    extern __shared__ float sdata[];

    // Pass 1: row max.
    float my_max = -INFINITY;
    for (int i = tid; i < n; i += bs) {
        float v = __half2float(x_row[i]);
        if (v > my_max) my_max = v;
    }
    sdata[tid] = my_max;
    __syncthreads();
    for (int stride = bs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = sdata[tid];
            float b = sdata[tid + stride];
            sdata[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    const float row_max = sdata[0];
    __syncthreads();

    // Pass 2: sum of exp(x - max).
    float my_sum = 0.0f;
    for (int i = tid; i < n; i += bs) {
        float v = __half2float(x_row[i]);
        my_sum += expf(v - row_max);
    }
    sdata[tid] = my_sum;
    __syncthreads();
    for (int stride = bs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    const float inv_denom = 1.0f / sdata[0];

    // Pass 3: write.
    for (int i = tid; i < n; i += bs) {
        float v = __half2float(x_row[i]);
        y_row[i] = __float2half(expf(v - row_max) * inv_denom);
    }
}

extern "C" hipError_t logismos_launch_softmax_fp16(
    const void* x_fp16,
    void*       y_fp16,
    int m,
    int n,
    hipStream_t stream)
{
    int bs = 256;
    if (n < bs) {
        int p = 1;
        while (p < n) p <<= 1;
        bs = p;
    }
    dim3 block(bs, 1, 1);
    dim3 grid(m, 1, 1);
    size_t lds_bytes = bs * sizeof(float);
    logismos_softmax_fp16_kernel<<<grid, block, lds_bytes, stream>>>(
        reinterpret_cast<const __half*>(x_fp16),
        reinterpret_cast<__half*>(y_fp16),
        m, n);
    return hipGetLastError();
}
