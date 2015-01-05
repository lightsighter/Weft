/*
 * Copyright 2015 Stanford University and NVIDIA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>

#include "cudaDMA.h"
#include "params.h"

/*
 * This version of saxpy uses cudaDMA for DMAs with manual double buffering.
 */
__global__ void saxpy_cudaDMA_doublebuffer ( float* y, float* x, float a, clock_t * timer_vals) 
{
  __shared__ float sdata_x0 [COMPUTE_THREADS_PER_CTA];
  __shared__ float sdata_x1 [COMPUTE_THREADS_PER_CTA];
  __shared__ float sdata_y0 [COMPUTE_THREADS_PER_CTA];
  __shared__ float sdata_y1 [COMPUTE_THREADS_PER_CTA];

  cudaDMASequential<true, 16, DMA_SZ, DMA_THREADS_PER_LD>
    dma_ld_x_0 (1, COMPUTE_THREADS_PER_CTA, COMPUTE_THREADS_PER_CTA);
  cudaDMASequential<true, 16, DMA_SZ, DMA_THREADS_PER_LD>
    dma_ld_y_0 (2, COMPUTE_THREADS_PER_CTA, COMPUTE_THREADS_PER_CTA + DMA_THREADS_PER_LD);
  cudaDMASequential<true, 16, DMA_SZ, DMA_THREADS_PER_LD>
    dma_ld_x_1 (3, COMPUTE_THREADS_PER_CTA, COMPUTE_THREADS_PER_CTA + 2*DMA_THREADS_PER_LD);
  cudaDMASequential<true, 16, DMA_SZ, DMA_THREADS_PER_LD>
    dma_ld_y_1 (4, COMPUTE_THREADS_PER_CTA, COMPUTE_THREADS_PER_CTA + 3*DMA_THREADS_PER_LD);

  int tid = threadIdx.x ;

  if ( tid < COMPUTE_THREADS_PER_CTA ) {
    unsigned int idx;
    int i;
    float tmp_x;
    float tmp_y;
    
    // Preamble:
    dma_ld_x_0.start_async_dma();
    dma_ld_y_0.start_async_dma();
    dma_ld_x_1.start_async_dma();
    dma_ld_y_1.start_async_dma();
    #pragma unroll 1
    for (i = 0; i < NUM_ITERS-2; i += 2) {
      
      // Phase 1:
      dma_ld_x_0.wait_for_dma_finish();
      tmp_x = sdata_x0[tid];
      dma_ld_x_0.start_async_dma();
      dma_ld_y_0.wait_for_dma_finish();
      tmp_y = sdata_y0[tid];
      dma_ld_y_0.start_async_dma();
      idx = i * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + threadIdx.x;
      y[idx] = a * tmp_x + tmp_y;

      // Phase 2:
      dma_ld_x_1.wait_for_dma_finish();
      tmp_x = sdata_x1[tid];
      dma_ld_x_1.start_async_dma();
      dma_ld_y_1.wait_for_dma_finish();
      tmp_y = sdata_y1[tid];
      dma_ld_y_1.start_async_dma();
      idx = (i+1) * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + threadIdx.x;
      y[idx] = a * tmp_x + tmp_y;
    }
      
    // Postamble
    dma_ld_x_0.wait_for_dma_finish();
    tmp_x = sdata_x0[tid];
    dma_ld_y_0.wait_for_dma_finish();
    tmp_y = sdata_y0[tid];
    idx = i * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + threadIdx.x;
    y[idx] = a * tmp_x + tmp_y;
    dma_ld_x_1.wait_for_dma_finish();
    tmp_x = sdata_x1[tid];
    dma_ld_y_1.wait_for_dma_finish();
    tmp_y = sdata_y1[tid];
    idx = (i+1) * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + threadIdx.x;
    y[idx] = a * tmp_x + tmp_y;

  } else if (dma_ld_x_0.owns_this_thread()) {
    #pragma unroll 1
    for (unsigned int j = 0; j < NUM_ITERS; j+=2) {
      // idx is a pointer to the base of the chunk of memory to copy
      unsigned int idx = j * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA;
      dma_ld_x_0.execute_dma( &x[idx], sdata_x0 );
    }
  } else if (dma_ld_y_0.owns_this_thread()) {
    #pragma unroll 1
    for (unsigned int j = 0; j < NUM_ITERS; j+=2) {
      unsigned int idx = j * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA;
      dma_ld_y_0.execute_dma( &y[idx], sdata_y0 );
    }
  } else if (dma_ld_x_1.owns_this_thread()) {
    #pragma unroll 1
    for (unsigned int j = 1; j < NUM_ITERS; j+=2) {
      unsigned int idx = j * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA;
      dma_ld_x_1.execute_dma( &x[idx], sdata_x1 );
    }
  } else if (dma_ld_y_1.owns_this_thread()) {
    #pragma unroll 1
    for (unsigned int j = 1; j < NUM_ITERS; j+=2) {
      unsigned int idx = j * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA;
      dma_ld_y_1.execute_dma( &y[idx], sdata_y1 );
    }
  }
}

