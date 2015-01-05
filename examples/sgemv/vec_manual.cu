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

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>

#include "cuda.h"
#include "cuda_runtime.h"

#include "cudaDMA.h"

#define SIZE_N	        896
#define SIZE_M		SIZE_N

#define DMA_KERNEL			sgemvn_cuda_dma_vec_manual
#define COMPUTE_THREADS_PER_CTA		128	
#define DMA_THREADS_PER_LD		32
#define DMA_LDS				1
#ifndef VEC_ELMTS
#define VEC_ELMTS			512	
#endif

#ifndef SGEMV_ITERS
#define SGEMV_ITERS                     128
#endif

__global__ void
__launch_bounds__(160,1)
sgemvn_cuda_dma_vec_manual(int n, int m, int n1, float alpha, float *A, int lda, float *x, float *y)
{
	__shared__ float buff0[VEC_ELMTS];
	__shared__ float buff1[VEC_ELMTS];


	cudaDMASequential<true,16,4*VEC_ELMTS,DMA_THREADS_PER_LD>
	  dma_ld_0(1,COMPUTE_THREADS_PER_CTA,COMPUTE_THREADS_PER_CTA);
	cudaDMASequential<true,16,4*VEC_ELMTS,DMA_THREADS_PER_LD>
	  dma_ld_1(2,COMPUTE_THREADS_PER_CTA,COMPUTE_THREADS_PER_CTA);

	if (threadIdx.x < COMPUTE_THREADS_PER_CTA)
	{
		dma_ld_0.start_async_dma();	
		dma_ld_1.start_async_dma();
		int ind = blockIdx.x*COMPUTE_THREADS_PER_CTA + threadIdx.x;

		A += ind;

		float res = 0.f;

#ifdef DYNAMIC
                #pragma unroll 1
		for(int i=0; i<n1; i += (VEC_ELMTS*2) )
#else
                for(int i=0; i<SGEMV_ITERS; i++)
#endif
		{
			dma_ld_0.wait_for_dma_finish();
			#pragma unroll
			for(int j=0; j < VEC_ELMTS; j++)
			{
				res+=A[0]*buff0[j];
				A+=lda;
			}
			dma_ld_0.start_async_dma();
			dma_ld_1.wait_for_dma_finish();
			#pragma unroll
			for (int j=0; j < VEC_ELMTS; j++)
			{
				res+=A[0]*buff1[j];
				A+=lda;
			}
			dma_ld_1.start_async_dma();
		}

		if (ind<n)
			y[ind] = alpha * res;
	}
	else if (dma_ld_0.owns_this_thread())
	{
#ifdef DYNAMIC
                #pragma unroll 1
		for (int i=0; i<n1; i += (VEC_ELMTS*2))
#else
                for (int i=0; i<SGEMV_ITERS; i++)
#endif
		{
			dma_ld_0.execute_dma(x,buff0);
			x += VEC_ELMTS;
			dma_ld_1.execute_dma(x,buff1);
			x += VEC_ELMTS;
		}	
		dma_ld_0.wait_for_dma_start();
		dma_ld_1.wait_for_dma_start();
	}
}

