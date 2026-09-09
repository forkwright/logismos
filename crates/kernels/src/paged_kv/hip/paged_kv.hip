#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>

namespace {
__device__ std::size_t offset(
    std::uint32_t layer,
    std::uint32_t page,
    std::uint32_t token,
    std::uint32_t column,
    std::uint32_t row_width,
    std::uint32_t page_tokens,
    std::uint32_t physical_pages) {
    return (((static_cast<std::size_t>(layer) * physical_pages + page) * page_tokens + token)
        * row_width) + column;
}
}

extern "C" __global__ void logismos_paged_kv_copy_tail_f32_kernel(
    float* keys,
    float* values,
    std::uint32_t layers,
    std::uint32_t row_width,
    std::uint32_t page_tokens,
    std::uint32_t physical_pages,
    std::uint32_t source_page,
    std::uint32_t destination_page,
    std::uint32_t filled_tokens) {
    const std::size_t total = static_cast<std::size_t>(layers) * 2U * filled_tokens * row_width;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t column = static_cast<std::uint32_t>(index % row_width);
        const std::size_t row = index / row_width;
        const std::uint32_t token = static_cast<std::uint32_t>(row % filled_tokens);
        const std::size_t layer_kind = row / filled_tokens;
        const std::uint32_t kind = static_cast<std::uint32_t>(layer_kind % 2U);
        const std::uint32_t layer = static_cast<std::uint32_t>(layer_kind / 2U);
        const std::size_t source = offset(layer, source_page, token, column, row_width, page_tokens, physical_pages);
        const std::size_t destination = offset(layer, destination_page, token, column, row_width, page_tokens, physical_pages);
        if (kind == 0U) {
            keys[destination] = keys[source];
        } else {
            values[destination] = values[source];
        }
    }
}

extern "C" __global__ void logismos_paged_kv_append_row_f32_kernel(
    float* keys,
    float* values,
    const float* source_key,
    const float* source_value,
    std::uint32_t layer,
    std::uint32_t physical_page,
    std::uint32_t in_page_token,
    std::uint32_t row_width,
    std::uint32_t page_tokens,
    std::uint32_t physical_pages) {
    for (std::size_t column = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         column < row_width;
         column += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const auto destination = offset(layer, physical_page, in_page_token,
            static_cast<std::uint32_t>(column), row_width, page_tokens, physical_pages);
        keys[destination] = source_key[column];
        values[destination] = source_value[column];
    }
}

extern "C" __global__ void logismos_paged_kv_write_table_u32_kernel(
    std::uint32_t* table,
    std::uint32_t logical_page,
    std::uint32_t physical_page) {
    if (threadIdx.x == 0U && blockIdx.x == 0U) {
        table[logical_page] = physical_page;
    }
}
