/*
 * findMultipleBestGPU.h
 *
 *  Created on: Mar 10, 2020
 *      Author: neville
 */



/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* This example demonstrates how to use the CUBLAS library
 * by scaling an array of floating-point values on the device
 * and comparing the result to the same operation performed
 * on the host.
 */

#ifndef FINDMULTIPLEBESTGPU_H_
#define FINDMULTIPLEBESTGPU_H_


void findMultipleBestGPUs(int &num_of_devices, int *device_ids) {
  // Find the best CUDA capable GPU device
  int current_device = 0;

  int device_count;
  checkCudaErrors(cudaGetDeviceCount(&device_count));
  typedef struct gpu_perf_t {
    uint64_t compute_perf;
    int device_id;
  } gpu_perf;

  gpu_perf *gpu_stats = (gpu_perf *)malloc(sizeof(gpu_perf) * device_count);

  cudaDeviceProp deviceProp;
  int devices_prohibited = 0;
  while (current_device < device_count) {
    cudaGetDeviceProperties(&deviceProp, current_device);

    // If this GPU is not running on Compute Mode prohibited,
    // then we can add it to the list
    int sm_per_multiproc;
    if (deviceProp.computeMode != cudaComputeModeProhibited) {
      if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
        sm_per_multiproc = 1;
      } else {
        sm_per_multiproc =
            _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
      }

      gpu_stats[current_device].compute_perf =
          (uint64_t)deviceProp.multiProcessorCount * sm_per_multiproc *
          deviceProp.clockRate;
      gpu_stats[current_device].device_id = current_device;

    } else {
      devices_prohibited++;
    }

    ++current_device;
  }
  if (devices_prohibited == device_count) {
    fprintf(stderr,
            "gpuGetMaxGflopsDeviceId() CUDA error:"
            " all devices have compute mode prohibited.\n");
    exit(EXIT_FAILURE);
  } else {
    gpu_perf temp_elem;
    // Sort the GPUs by highest compute perf.
    for (int i = 0; i < current_device - 1; i++) {
      for (int j = 0; j < current_device - i - 1; j++) {
        if (gpu_stats[j].compute_perf < gpu_stats[j + 1].compute_perf) {
          temp_elem = gpu_stats[j];
          gpu_stats[j] = gpu_stats[j + 1];
          gpu_stats[j + 1] = temp_elem;
        }
      }
    }

    for (int i = 0; i < num_of_devices; i++) {
      device_ids[i] = gpu_stats[i].device_id;
    }
  }
  free(gpu_stats);
}





#endif /* FINDMULTIPLEBESTGPU_H_ */
