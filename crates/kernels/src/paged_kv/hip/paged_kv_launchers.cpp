#include <cstddef>
#include <cstdint>
#include <limits>
#include <hip/hip_runtime.h>

extern "C" __global__ void logismos_paged_kv_copy_tail_f32_kernel(
    float*, float*, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t);
extern "C" __global__ void logismos_paged_kv_append_row_f32_kernel(
    float*, float*, const float*, const float*, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t);
extern "C" __global__ void logismos_paged_kv_write_table_u32_kernel(
    std::uint32_t*, std::uint32_t, std::uint32_t);

namespace {
constexpr std::uint32_t THREADS = 256;

std::uint32_t blocks_for(std::size_t elements) {
    const auto blocks = (elements + THREADS - 1U) / THREADS;
    const auto capped = blocks > std::numeric_limits<std::uint32_t>::max()
        ? std::numeric_limits<std::uint32_t>::max()
        : blocks;
    return static_cast<std::uint32_t>(capped > 0U ? capped : 1U);
}
}

extern "C" hipError_t logismos_launch_paged_kv_copy_tail_f32(
    void* keys, void* values, std::uint32_t layers, std::uint32_t row_width,
    std::uint32_t page_tokens, std::uint32_t physical_pages,
    std::uint32_t source_page, std::uint32_t destination_page,
    std::uint32_t filled_tokens, hipStream_t stream) {
    const auto elements = static_cast<std::size_t>(layers) * 2U * filled_tokens * row_width;
    logismos_paged_kv_copy_tail_f32_kernel<<<blocks_for(elements), THREADS, 0, stream>>>(
        static_cast<float*>(keys), static_cast<float*>(values), layers, row_width,
        page_tokens, physical_pages, source_page, destination_page, filled_tokens);
    return hipGetLastError();
}

extern "C" hipError_t logismos_launch_paged_kv_append_row_f32(
    void* keys, void* values, const void* source_key, const void* source_value,
    std::uint32_t layer, std::uint32_t physical_page, std::uint32_t in_page_token,
    std::uint32_t row_width, std::uint32_t page_tokens, std::uint32_t physical_pages,
    hipStream_t stream) {
    logismos_paged_kv_append_row_f32_kernel<<<blocks_for(row_width), THREADS, 0, stream>>>(
        static_cast<float*>(keys), static_cast<float*>(values), static_cast<const float*>(source_key),
        static_cast<const float*>(source_value), layer, physical_page, in_page_token,
        row_width, page_tokens, physical_pages);
    return hipGetLastError();
}

extern "C" hipError_t logismos_launch_paged_kv_write_table_u32(
    void* table, std::uint32_t logical_page, std::uint32_t physical_page, hipStream_t stream) {
    logismos_paged_kv_write_table_u32_kernel<<<1, 1, 0, stream>>>(
        static_cast<std::uint32_t*>(table), logical_page, physical_page);
    return hipGetLastError();
}
