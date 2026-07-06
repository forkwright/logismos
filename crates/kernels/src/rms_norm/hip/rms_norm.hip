// RMSNorm kernel — one workgroup per row, LDS-reduce sum-of-squares.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

extern "C" __global__ void logismos_rms_norm_fp16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    __half*       __restrict__ y,
    int m,
    int n,
    float eps)
{
    // One block per row.
    int row = blockIdx.x;
    if (row >= m) return;

    const __half* x_row = x + row * n;
    __half*       y_row = y + row * n;

    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    // Stage 1: thread-local partial sum of squares.
    float partial = 0.0f;
    for (int i = tid; i < n; i += bs) {
        float v = __half2float(x_row[i]);
        partial += v * v;
    }

    // Stage 2: block-reduction via LDS. Handles arbitrary bs up to
    // 1024; no warp-shuffle here to keep the kernel dead simple and
    // portable across wave32 / wave64 execution — wave32 on gfx1100 is
    // the target but the reduction shape is identical either way.
    extern __shared__ float sdata[];
    sdata[tid] = partial;
    __syncthreads();

    for (int stride = bs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    const float sum_sq = sdata[0];
    const float mean_sq = sum_sq / (float)n;
    const float inv_rms = rsqrtf(mean_sq + eps);

    // Stage 3: scaled write-back.
    for (int i = tid; i < n; i += bs) {
        float v = __half2float(x_row[i]);
        float scale = __half2float(w[i]);
        y_row[i] = __float2half(v * inv_rms * scale);
    }
}

extern "C" hipError_t logismos_launch_rms_norm_fp16(
    const void* x_fp16,
    const void* w_fp16,
    void*       y_fp16,
    int m,
    int n,
    float eps,
    hipStream_t stream)
{
    // Use up to 256 threads; works with arbitrary n.
    int bs = 256;
    if (n < bs) {
        // Round up to next power of two, capped at 256.
        int p = 1;
        while (p < n) p <<= 1;
        bs = p;
    }
    dim3 block(bs, 1, 1);
    dim3 grid(m, 1, 1);
    size_t lds_bytes = bs * sizeof(float);
    logismos_rms_norm_fp16_kernel<<<grid, block, lds_bytes, stream>>>(
        reinterpret_cast<const __half*>(x_fp16),
        reinterpret_cast<const __half*>(w_fp16),
        reinterpret_cast<__half*>(y_fp16),
        m, n, eps);
    return hipGetLastError();
}
