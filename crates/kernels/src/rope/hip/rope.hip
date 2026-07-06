// RoPE kernel, in-place on fp16 (B, S, H, D). Interleaved cos/sin
// table layout matches the CPU reference.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

extern "C" __global__ void logismos_rope_fp16_kernel(
    __half* __restrict__ qk,
    const float* __restrict__ cos_sin,
    int batch,
    int seq,
    int heads,
    int head_dim)
{
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;  // 0 .. head_dim/2
    int h        = blockIdx.y;                              // 0 .. heads
    int s        = blockIdx.z % seq;                        // 0 .. seq
    int b        = blockIdx.z / seq;                        // 0 .. batch

    const int pairs = head_dim / 2;
    if (pair_idx >= pairs || h >= heads || s >= seq || b >= batch) return;

    const float* row = cos_sin + s * head_dim;
    float cos_v = row[2 * pair_idx];
    float sin_v = row[2 * pair_idx + 1];

    int base = ((b * seq + s) * heads + h) * head_dim;
    float x0 = __half2float(qk[base + 2 * pair_idx]);
    float x1 = __half2float(qk[base + 2 * pair_idx + 1]);
    float y0 = x0 * cos_v - x1 * sin_v;
    float y1 = x0 * sin_v + x1 * cos_v;
    qk[base + 2 * pair_idx]     = __float2half(y0);
    qk[base + 2 * pair_idx + 1] = __float2half(y1);
}

extern "C" hipError_t logismos_launch_rope_fp16(
    void* qk_fp16,
    const void* cos_sin_f32,
    int batch,
    int seq,
    int heads,
    int head_dim,
    hipStream_t stream)
{
    const int pairs = head_dim / 2;
    int bs = 64;
    if (pairs < bs) {
        int p = 1;
        while (p < pairs) p <<= 1;
        bs = p < 1 ? 1 : p;
    }
    dim3 block(bs, 1, 1);
    dim3 grid((pairs + bs - 1) / bs, heads, batch * seq);
    logismos_rope_fp16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__half*>(qk_fp16),
        reinterpret_cast<const float*>(cos_sin_f32),
        batch, seq, heads, head_dim);
    return hipGetLastError();
}
