#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

#include "numerical_status_bits.h"

__device__ inline bool logismos_status_is_subnormal(float value)
{
    constexpr std::uint32_t exponent_mask = 0x7f800000U;
    constexpr std::uint32_t significand_mask = 0x007fffffU;
    const std::uint32_t bits = __float_as_uint(value);
    return (bits & exponent_mask) == 0U && (bits & significand_mask) != 0U;
}

__device__ inline bool logismos_status_is_nonfinite(float value)
{
    constexpr std::uint32_t exponent_mask = 0x7f800000U;
    return (__float_as_uint(value) & exponent_mask) == exponent_mask;
}

__device__ inline void logismos_status_or(std::uint32_t* status, std::uint32_t bit)
{
    if (status != nullptr) {
        atomicOr(status, bit);
    }
}

__device__ inline float logismos_status_input(float value, std::uint32_t* status)
{
    if (logismos_status_is_subnormal(value)) {
        logismos_status_or(status, LOGISMOS_NUMERICAL_STATUS_INPUT_SUBNORMAL);
    }
    if (logismos_status_is_nonfinite(value)) {
        logismos_status_or(status, LOGISMOS_NUMERICAL_STATUS_INPUT_NONFINITE);
    }
    return value;
}

__device__ inline void logismos_status_f16_input_bits(std::uint16_t bits, std::uint32_t* status)
{
    constexpr std::uint16_t exponent_mask = 0x7c00U;
    constexpr std::uint16_t significand_mask = 0x03ffU;
    if ((bits & exponent_mask) == 0U && (bits & significand_mask) != 0U) {
        logismos_status_or(status, LOGISMOS_NUMERICAL_STATUS_INPUT_SUBNORMAL);
    }
    if ((bits & exponent_mask) == exponent_mask) {
        logismos_status_or(status, LOGISMOS_NUMERICAL_STATUS_INPUT_NONFINITE);
    }
}

__device__ inline float logismos_status_result(float value, std::uint32_t* status)
{
    if (logismos_status_is_subnormal(value)) {
        logismos_status_or(status, LOGISMOS_NUMERICAL_STATUS_ARITHMETIC_SUBNORMAL);
    }
    if (logismos_status_is_nonfinite(value)) {
        logismos_status_or(status, LOGISMOS_NUMERICAL_STATUS_ARITHMETIC_NONFINITE);
    }
    return value;
}

__device__ inline float logismos_status_arithmetic_operand(float value, std::uint32_t* status)
{
    return logismos_status_result(value, status);
}

__device__ inline float logismos_status_add(float left, float right, std::uint32_t* status)
{
    logismos_status_arithmetic_operand(left, status);
    logismos_status_arithmetic_operand(right, status);
    const float result = left + right;
    logismos_status_result(result, status);
    return result;
}

__device__ inline float logismos_status_sub(float left, float right, std::uint32_t* status)
{
    logismos_status_arithmetic_operand(left, status);
    logismos_status_arithmetic_operand(right, status);
    const float result = left - right;
    logismos_status_result(result, status);
    return result;
}

__device__ inline float logismos_status_mul(float left, float right, std::uint32_t* status)
{
    logismos_status_arithmetic_operand(left, status);
    logismos_status_arithmetic_operand(right, status);
    const float result = left * right;
    logismos_status_result(result, status);
    return result;
}

__device__ inline float logismos_status_div(float left, float right, std::uint32_t* status)
{
    logismos_status_arithmetic_operand(left, status);
    logismos_status_arithmetic_operand(right, status);
    const float result = left / right;
    logismos_status_result(result, status);
    return result;
}

__device__ inline float logismos_status_exp(float value, std::uint32_t* status)
{
    logismos_status_arithmetic_operand(value, status);
    const float result = expf(value);
    logismos_status_result(result, status);
    return result;
}

__device__ inline float logismos_status_sqrt(float value, std::uint32_t* status)
{
    logismos_status_arithmetic_operand(value, status);
    const float result = sqrtf(value);
    logismos_status_result(result, status);
    return result;
}

__device__ inline float logismos_status_rsqrt(float value, std::uint32_t* status)
{
    logismos_status_arithmetic_operand(value, status);
    const float result = rsqrtf(value);
    logismos_status_result(result, status);
    return result;
}

__device__ inline float logismos_status_log1p(float value, std::uint32_t* status)
{
    logismos_status_arithmetic_operand(value, status);
    const float result = log1pf(value);
    logismos_status_result(result, status);
    return result;
}
