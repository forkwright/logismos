/*
 * HIP runtime-API wrapper header. Included by bindgen (build.rs).
 *
 * Runtime API only (hipMalloc / hipMemcpy / hipStream* / hipEvent*
 * / hipGetDeviceProperties / hipDeviceSynchronize). Driver API and
 * hipBLASLt are out of scope for Phase 1.
 */
#ifndef LOGISMOS_HIPCORE_WRAPPER_RUNTIME_H
#define LOGISMOS_HIPCORE_WRAPPER_RUNTIME_H

#define __HIP_PLATFORM_AMD__ 1

#include <hip/hip_runtime_api.h>
#include <hip/driver_types.h>

#endif
