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
#include <assert.h>
#include <math.h>

#include "cudaDMAK.h"

#define LDG_BYTES	(12*16)

#ifndef USE_REDUCTION
#define USE_REDUCTION 0
#endif

#ifndef X_BEFORE_Y_FOR_PXY
#define X_BEFORE_Y_FOR_PXY 1
#endif

#define Z_BEFORE_X_FOR_PXZ 1
#define Z_BEFORE_Y_FOR_PYZ 1

#ifndef R
#define R	4
#endif

#ifndef USE_CONSTANT_FOR_ZCOEFF
#define USE_CONSTANT_FOR_ZCOEFF 1
#endif

#ifndef TILE_X
#define	TILE_X 32	
#endif
#ifndef TILE_Y
#define	TILE_Y	4		
#endif

#ifndef BLOCKING_TYPE
#define BLOCKING_TYPE float2
#endif
const int x_b = sizeof(BLOCKING_TYPE)/sizeof(float);
// Need radius in X-dimension to be a multiple of the X-dimension blocking factor
const int RX = (R % x_b ? R + x_b - (R % x_b) : R);

#define FP_RAND() (float(rand())/float(RAND_MAX) < 0.5 ? -(float((rand()+1)%5)) : (float((rand()+1)%5)))
#define FP_RAND_RAD() (float((rand()%4)) * M_PI_2)

#ifndef MAX_Z_DIM
#define MAX_Z_DIM 500
#endif

#ifndef RTM_ELMTS
#define RTM_ELMTS 64
#endif

__constant__ float c_xx[R+1], c_yy[R+1];
__constant__ float c_x[R+1], c_y[R+1];
__constant__ float vsz2_constant;

#ifdef USE_CONSTANT_FOR_ZCOEFF
__constant__ float c_zz[(2*R+1)*MAX_Z_DIM], c_z[(2*R+1)*MAX_Z_DIM];
#endif

template<class T, unsigned ecnt>
__device__ __forceinline__ void
shift(T b[ecnt+1]) {
#pragma unroll
    for(unsigned idx=0;idx<ecnt;idx++)
        b[idx] = b[idx+1];
}

template<class T, unsigned ecnt>
__device__ inline void
init(float b[ecnt+1], const T ival) {
#pragma unroll
    for(unsigned idx = 0; idx < ecnt + 1; idx ++)
        b[idx] = ival;
}

template<class T>
struct deriv_t {
    T dx2;
    T dy2;
    T dz2;
    T dxy;
    T dxz;
    T dyz;
};

struct range {
    int start;
    int end;
    range(int s, int e) : start(s), end(e) {}
};

enum Param_type_e {
    Vpz2,
    Delta,
    Epsln,
    Alpha,
    Beta,
    ParameterCnt
};

#define REGS_PER_THREAD 64
#define REGS_PER_SM 65536

__device__ __forceinline__ float
cachedRead(const float* data, const int index)
{
    const float* address = &data[index];
    float result;
    asm("ld.ca.f32 %0, [%1];\n"
        : "=f" (result)
#if defined(_WIN64) || defined(__LP64__)
        : "l"(address)
#else
        : "r"(address)
#endif
        : "memory"
       );
    return result;
}

__device__ __forceinline__ void
_myRedAdd(const float* address, const float update)
{
    asm("red.global.add.f32 [%0], %1;\n"
        :
#if defined(_WIN64) || defined(__LP64__)
        : "l"(address)
#else
        : "r"(address)
#endif
          , "f" (update)
        : "memory"
        );
}

#define DIV_CEILING(x,y) (x/y + (x % y ? 1 : 0))
#define EPT(x) (DIV_CEILING(2*R+x, x))
#define WARP_WIDTH 32
#define WPAD 0
#define MAX(a, b) (a < b ? b : a)
#define PQW_WIDTH MAX(WARP_WIDTH+WPAD, tile_x+2*R+WPAD)
#define halfWarpCnt (TILE_Y * TILE_X / WARP_WIDTH)
#define haloCnt ((2*R)/TILE_Y + 1)
#ifdef USE_TEX
texture<float2, 1, cudaReadModeElementType> tex_PQ2;
texture<float2, 1, cudaReadModeElementType> tex_PnQn2;
texture<float, 1, cudaReadModeElementType> tex_PQ;
texture<float, 1, cudaReadModeElementType> tex_PnQn;
texture<float4, 1, cudaReadModeElementType> tex_abde;
texture<float, 1, cudaReadModeElementType> tex_vpz2;
texture<float, 1, cudaReadModeElementType> tex_P;
texture<float, 1, cudaReadModeElementType> tex_Pn;
texture<float, 1, cudaReadModeElementType> tex_Q;
texture<float, 1, cudaReadModeElementType> tex_Qn;
#endif
#ifdef USE_TEX
#define ld_PQ2_ro(_loc) tex1Dfetch(tex_PQ2, _loc)
#define ld_PnQn2_ro(_loc) tex1Dfetch(tex_PnQn2, _loc)
#define ld_PQ_ro(_loc) tex1Dfetch(tex_PQ, _loc)
#define ld_P_ro(_loc) tex1Dfetch(tex_P, _loc)
#define ld_Q_ro(_loc) tex1Dfetch(tex_Q, _loc)
#define ld_PQ_ro_cached(_loc) tex1Dfetch(tex_PQ, _loc)
#define ld_PnQn_ro(_loc) tex1Dfetch(tex_PnQn, _loc)
#define ld_Pn_ro(_loc) tex1Dfetch(tex_Pn, _loc)
#define ld_Qn_ro(_loc) tex1Dfetch(tex_Qn, _loc)
#define ld_param_ro(_loc, _param) tex1Dfetch(tex_##_param, _loc)
#else
#define ld_PQ2_ro(_loc) g_PQ[_loc] 
#define ld_PnQn2_ro(_loc) g_PnQn[_loc] 
#define ld_PQ_ro(_loc) g_PQ[_loc]
#define ld_P_ro(_loc) g_P[_loc]
#define ld_Q_ro(_loc) g_Q[_loc]
#if NVCC_SUPPORTS_UNROLL_INLINE_ASM==1
#define ld_PQ_ro_cached(_loc) cachedRead(g_PQ, _loc)
#define ld_P_ro_cached(_loc) cachedRead(g_P, _loc)
#define ld_Q_ro_cached(_loc) cachedRead(g_Q, _loc)
#else
#define ld_PQ_ro_cached(_loc) ld_PQ_ro(_loc)
#define ld_P_ro_cached(_loc) ld_P_ro(_loc)
#define ld_Q_ro_cached(_loc) ld_Q_ro(_loc)
#endif
#define ld_PnQn_ro(_loc) g_PnQn[_loc]
#define ld_Pn_ro(_loc) g_Pn[_loc]
#define ld_Qn_ro(_loc) g_Qn[_loc]
#define ld_param_ro(_loc, _param) g_##_param[_loc]
#endif

#define ZPENCIL_LENGTH (2*R+1)
#define ZPENCIL_LAST (ZPENCIL_LENGTH-1)
#define ZPENCIL_FIRST 0
#define ZPENCIL(_n, _t) _t _n[ZPENCIL_LENGTH]
#define ZPENCIL_SHIFT(_n) shift<float, ZPENCIL_LENGTH-1>(_n)
#define ZPENCIL_INIT(_n) init<float, ZPENCIL_LENGTH-1>(_n, 0.0f)
#define ZPENCIL_CTR_PRESHIFT R+1
#define ZPENCIL_CTR_POSTSHIFT R

#define Q_LENGTH R
#ifdef Q_IN_REGISTERS
#define Q_DEF(_n, _t) _t _n[Q_LENGTH]; init<float, Q_LENGTH-1>(_n, 0.0f)
#define qidx 
#define Q_COMMON(_i)
#define Q_CURR(_q, _i) _q[0]
#define Q_LAST(_q, _i) _q[Q_LENGTH-1]
#define ADVANCE_Qs(_q1,_q2,_q3,_i) shift<float, Q_LENGTH-1>(_q1);shift<float, Q_LENGTH-1>(_q2);shift<float, Q_LENGTH-1>(_q3)
#define ADVANCE_6Qs(_q1,_q2,_q3,_q4, _q5, _q6, _i) shift<float, Q_LENGTH-1>(_q1);shift<float, Q_LENGTH-1>(_q2);shift<float, Q_LENGTH-1>(_q3);shift<float, Q_LENGTH-1>(_q4);shift<float, Q_LENGTH-1>(_q5);shift<float, Q_LENGTH-1>(_q6)
#define ADVANCE_3Qs(_q1,_q2,_q3,_i) shift<float, Q_LENGTH-1>(_q1);shift<float,Q_LENGTH-1>(_q2);shift<float, Q_LENGTH-1>(_q3);
#else
#define Q_COMMON(_i) int _i = 0
#ifdef Q_IN_SMEM
#if 0
#define Q_DEF(_n, _t) volatile __shared__ _t _n[Q_LENGTH][1*tile_y][tile_x]
#define Q_CURR(_q, _i) _q[_i][threadIdx.y][threadIdx.x]
#define Q_LAST(_q, _i) _q[(_i == 0 ? Q_LENGTH-1 : _i-1)][threadIdx.y][threadIdx.x]
#endif
#define Q_DEF(_n, _t) volatile __shared__ _t _n[Q_LENGTH][TILE_Y*TILE_X]
#define Q_CURR(_q, _i) _q[_i][threadIdx.x]
#define Q_LAST(_q, _i) _q[(_i == 0 ? Q_LENGTH-1 : _i-1)][threadIdx.x]
#define ADVANCE_Qs(_q1,_q2,_q3,_i) _i = (_i == Q_LENGTH-1 ? 0 : _i+1)
#define ADVANCE_6Qs(_q1,_q2,_q3,_q4,_q5,_q6,_i) _i = (_i == Q_LENGTH-1 ? 0 : _i+1)
#else
#define Q_DEF(_n, _t) _t _n[Q_LENGTH]
#define Q_CURR(_q, _i) _q[_i]
#define Q_LAST(_q, _i) _q[(_i == 0 ? Q_LENGTH-1 : _i-1)]
#define ADVANCE_Qs(_q1,_q2,_q3, _i) _i = (_i == Q_LENGTH-1 ? 0 : _i+1)
#define ADVANCE_6Qs(_q1,_q2,_q3,_q4,_q5,_q6,_i) _i = (_i == Q_LENGTH-1 ? 0 : _i+1)
#endif
#endif

#define SMEM_Q_DEF(_n, _t) volatile __shared__ _t _n[Q_LENGTH][TILE_Y*TILE_X]
#define SMEM_Q_CURR(_q, _i) _q[_i][threadIdx.x]
#define SMEM_Q_LAST(_q, _i) _q[(_i == 0 ? Q_LENGTH-1 : _i-1)][threadIdx.x]
#define SMEM_ADVANCE_Qs(_q1,_q2,_q3,_i) _i = (_i == Q_LENGTH-1 ? 0 : _i+1)
#define SMEM_ADVANCE_6Qs(_q1,_q2,_q3,_q4,_q5,_q6,_i) _i = (_i == Q_LENGTH-1 ? 0 : _i+1)
#define SMEM_ADVANCE_3Qs(_q1,_q2,_q3,_i) _i = (_i == Q_LENGTH-1 ? 0 : _i+1)

#if NVCC_SUPPORTS_UNROLL_INLINE_ASM==1
#define READ_Z_COEFF(_a, _i) cachedRead(_a, _i)
#else
#define READ_Z_COEFF(_a, _i) _a[_i]
#endif

#define SMEM_ROW_WIDTH (2*(tile_x+2*R))

#if R <= TILE_Y
#define HALO_CNT 1
#define HALO_INIT {0.f}
#define LAST_HIDX 0
#else
#if R <= 2*TILE_Y
#define HALO_CNT 2
#define HALO_INIT {0.f, 0.f}
#define LAST_HIDX 1
#else
#error "Not coded to handle the case when R > 2*TILE_Y"
#endif
#endif

#if USE_REDUCTION == 1
#define REDUCTION_LD(_mv_ld_statement)
// _mv = memory value, _uv = update value
#define REDUCTION_SUB(_l, _mv, _uv) _myRedAdd(&_l, -(_uv))
#else
#define REDUCTION_LD(_mv_ld_statement) _mv_ld_statement
#define REDUCTION_SUB(_l, _mv, _uv) _l = (_mv) - (_uv)
#endif

#define COMPUTE_THREADS_PER_CTA (TILE_X*TILE_Y)
#define COUNT_PER_STRIDE_ROW	(TILE_X+2*R) // The extra 32 covers the 2*R halo elements
#define DMA_THREADS_PER_LD 		32	
//#define DMA_THREADS_PER_LD 		64	
//#define DMA_THREADS_PER_LD		128
//#define DMA_THREADS_PER_LD		256	
#define DMA_THREADS_PER_CTA		(1*DMA_THREADS_PER_LD)
#define BYTES_PER_THREAD		(sizeof(float2)*COUNT_PER_STRIDE_ROW*(TILE_Y+2*R)/DMA_THREADS_PER_LD)

#define PQY_BUF_OFFSET			(-(R*COUNT_PER_STRIDE_ROW))

__global__ void
__launch_bounds__(160,3)
single_pass_ktuned_DMA_single_specialized_one_phase(float2 *g_PnQn,
                   float2* g_PQ, 
#ifndef USE_TEX				   
				   float4* g_abde, float* g_vpz2,
#endif
                   const int row_stride, const int slice_stride, const int nz,
                   const int R_x_row_stride, const int tile_y_x_row_stride,
                   const int Pn_P_diff, const int lead_pad, const int offset, 
                   const int q_start_idx
#ifndef USE_CONSTANT_FOR_ZCOEFF
                   , const float* c_z, const float* c_zz
#endif
    )
{
	int tid = threadIdx.x;
	__shared__ float2 s_PQ[(TILE_Y+2*R)*COUNT_PER_STRIDE_ROW+TILE_Y*COUNT_PER_STRIDE_ROW];
	float2 *PQy_buf = s_PQ+(TILE_Y+2*R)*COUNT_PER_STRIDE_ROW+PQY_BUF_OFFSET;
	cudaDMAStrided<true,16,LDG_BYTES,COUNT_PER_STRIDE_ROW*sizeof(float2),DMA_THREADS_PER_LD,TILE_Y+2*R>
		dma_ld_pq(1,
				COMPUTE_THREADS_PER_CTA,
				COMPUTE_THREADS_PER_CTA,
				row_stride*sizeof(float2),
				COUNT_PER_STRIDE_ROW*sizeof(float2)
				);
	int gid = offset + blockIdx.x*TILE_X +
					 + blockIdx.y*TILE_Y*row_stride;

	if(tid<COMPUTE_THREADS_PER_CTA)
	{
		// Compute 2D index
		const int tx = tid % TILE_X;
		const int ty = tid / TILE_X;

		int out_idx = gid + ty*row_stride + tx;

		// SMEM pointers
		const int PQ_index = (R+ty)*COUNT_PER_STRIDE_ROW + (R+tx);

		// Halo work for y stencil
		// R=4 ONLY
		const bool Xlower = (tx<(WARP_WIDTH/2));
		const int Xofs = (tx & 0x3) + 1;
		const int Xalign = (tx & 0xf);
		const int PQ_halo_diff = (Xlower ?
								 -(Xalign+Xofs) :
								  (WARP_WIDTH/2) - (Xalign+1) + Xofs);
		// State
		ZPENCIL(l_P, float);
		ZPENCIL(l_Px,float);
		ZPENCIL(l_Py,float);
		ZPENCIL(l_Q, float);
		ZPENCIL(l_Qx,float);
		ZPENCIL(l_Qy,float);
		ZPENCIL_INIT(l_P);
		ZPENCIL_INIT(l_Px);
		ZPENCIL_INIT(l_Py);
		ZPENCIL_INIT(l_Q);
		ZPENCIL_INIT(l_Qx);
		ZPENCIL_INIT(l_Qy);

		Q_COMMON(qidx);
		Q_DEF (q_Pxx, float);
		Q_DEF (q_Pyy, float);
		Q_DEF (q_Pxy, float);
		Q_DEF (q_Qxx, float);
		Q_DEF (q_Qyy, float);
		Q_DEF (q_Qxy, float);

		//------------------------------------------------------
		// Prime the pump
		//------------------------------------------------------
		dma_ld_pq.start_async_dma();
		for(int i=0; i<2*R; i++)
		{
			ZPENCIL_SHIFT(l_P);
			ZPENCIL_SHIFT(l_Px);
			ZPENCIL_SHIFT(l_Py);
			ZPENCIL_SHIFT(l_Q);
			ZPENCIL_SHIFT(l_Qx);
			ZPENCIL_SHIFT(l_Qy);
			ADVANCE_6Qs(q_Pxx, q_Pyy, q_Pxy, q_Qxx, q_Qyy, q_Qxy, qidx);

			dma_ld_pq.wait_for_dma_finish(); // s_PQ ready, synchronization point for compute threads
			l_P[ZPENCIL_LAST] = s_PQ[PQ_index].x;
			l_Q[ZPENCIL_LAST] = s_PQ[PQ_index].y;

			// Compute x, xx stencils
			l_Px[ZPENCIL_LAST] = c_x[0]*l_P[ZPENCIL_LAST];
        	l_Qx[ZPENCIL_LAST] = c_x[0]*l_Q[ZPENCIL_LAST];
        	float q_Pxxl = c_xx[0]*l_P[ZPENCIL_LAST];
        	float q_Qxxl = c_xx[0]*l_Q[ZPENCIL_LAST];
			#pragma unroll
			for(int j=1;j<=R;j++) {
				const float2 v1 = s_PQ[PQ_index-j];
				const float2 v2 = s_PQ[PQ_index+j];
				l_Px[ZPENCIL_LAST] += c_x[j] * (v1.x + v2.x);
				l_Qx[ZPENCIL_LAST] += c_x[j] * (v1.y + v2.y);
				q_Pxxl += c_xx[j] * (v1.x + v2.x);
				q_Qxxl += c_xx[j] * (v1.y + v2.y);
			}
			Q_LAST(q_Pxx, qidx) = q_Pxxl;
			Q_LAST(q_Qxx, qidx) = q_Qxxl;

			// Compute y stencils for halo
			float2 PQyside;
			PQyside.x = c_y[0] * s_PQ[PQ_index+PQ_halo_diff].x; 
			PQyside.y = c_y[0] * s_PQ[PQ_index+PQ_halo_diff].y; 
			#pragma unroll
			for(int j=1;j<=R;j++)
			{
				const float2 v1 = s_PQ[PQ_index+PQ_halo_diff-j*COUNT_PER_STRIDE_ROW];
				const float2 v2 = s_PQ[PQ_index+PQ_halo_diff+j*COUNT_PER_STRIDE_ROW];
				PQyside.x += c_y[j] * (v1.x + v2.x);
				PQyside.y += c_y[j] * (v1.y + v2.y);
			}
			PQy_buf[PQ_index+PQ_halo_diff] = PQyside;

			// Compute y, yy stencils
        	l_Py[ZPENCIL_LAST] = c_y[0]*l_P[ZPENCIL_LAST];
        	l_Qy[ZPENCIL_LAST] = c_y[0]*l_Q[ZPENCIL_LAST];
        	float q_Pyyl = c_yy[0]*l_P[ZPENCIL_LAST];
        	float q_Qyyl = c_yy[0]*l_Q[ZPENCIL_LAST];
			#pragma unroll
			for(int j=1;j<=R;j++) {
				const float2 v1 = s_PQ[PQ_index-j*COUNT_PER_STRIDE_ROW];
				const float2 v2 = s_PQ[PQ_index+j*COUNT_PER_STRIDE_ROW];
				l_Py[ZPENCIL_LAST] += c_y[j] * (v1.x + v2.x);
				l_Qy[ZPENCIL_LAST] += c_y[j] * (v1.y + v2.y);
				q_Pyyl += c_yy[j] * (v1.x + v2.x);
				q_Qyyl += c_yy[j] * (v1.y + v2.y);
			}
			Q_LAST(q_Pyy, qidx) = q_Pyyl;
			Q_LAST(q_Qyy, qidx) = q_Qyyl;
			PQy_buf[PQ_index].x = l_Py[ZPENCIL_LAST];
			PQy_buf[PQ_index].y = l_Qy[ZPENCIL_LAST];

			ptx_cudaDMA_barrier_blocking(0,COMPUTE_THREADS_PER_CTA); // __syncthreads()
			dma_ld_pq.start_async_dma(); // fetch next s_PQ

			// Compute xy 
			float q_Pxyl = c_x[0]*l_Py[ZPENCIL_LAST];
			float q_Qxyl = c_x[0]*l_Qy[ZPENCIL_LAST];
			#pragma unroll
			for(int j=1;j<=R;j++)
			{
				const float2 v1 = PQy_buf[PQ_index-j];
				const float2 v2 = PQy_buf[PQ_index+j];
				q_Pxyl += c_x[j] * (v1.x + v2.x);
				q_Qxyl += c_x[j] * (v1.y + v2.y);
			}
			Q_LAST(q_Pxy, qidx) = q_Pxyl;
			Q_LAST(q_Qxy, qidx) = q_Qxyl;
		}

		//------------------------------------------------------
		// Main loop
		//------------------------------------------------------
		int z_base = 0;
#ifdef DYNAMIC
		for(int iz=0; iz<(nz-1); iz++, z_base+=(2*R+1))
#else
                #pragma unroll 1
                for (int iz=0; iz<(RTM_ELMTS-1); iz++, z_base+=(2*R+1))
#endif
		{
            ZPENCIL_SHIFT(l_P);
            ZPENCIL_SHIFT(l_Px);
            ZPENCIL_SHIFT(l_Py);
            ZPENCIL_SHIFT(l_Q);
            ZPENCIL_SHIFT(l_Qx);
            ZPENCIL_SHIFT(l_Qy);
            const float q_Pxx0 = Q_CURR(q_Pxx, qidx);
            const float q_Qxx0 = Q_CURR(q_Qxx, qidx);
            const float q_Pyy0 = Q_CURR(q_Pyy, qidx);
            const float q_Qyy0 = Q_CURR(q_Qyy, qidx);
            const float q_Pxy0 = Q_CURR(q_Pxy, qidx);
            const float q_Qxy0 = Q_CURR(q_Qxy, qidx);
            ADVANCE_6Qs(q_Pxx, q_Pyy, q_Pxy, q_Qxx, q_Qyy, q_Qxy, qidx);

			// Params
			REDUCTION_LD(const float2 PnQn = ld_PnQn2_ro(out_idx));
			const float4 abde = ld_param_ro(out_idx, abde);
			const float  vpz2 = ld_param_ro(out_idx, vpz2);

			dma_ld_pq.wait_for_dma_finish(); // s_PQ ready, synchronization point for compute threads

			l_P[ZPENCIL_LAST] = s_PQ[PQ_index].x;
			l_Q[ZPENCIL_LAST] = s_PQ[PQ_index].y;

			// Compute x, xx stencils
            l_Px[ZPENCIL_LAST] = c_x[0]*l_P[ZPENCIL_LAST];
            l_Qx[ZPENCIL_LAST] = c_x[0]*l_Q[ZPENCIL_LAST];
            float q_Pxxl = c_xx[0]*l_P[ZPENCIL_LAST];
            float q_Qxxl = c_xx[0]*l_Q[ZPENCIL_LAST];
			#pragma unroll
            for(int i=1;i<=R;i++) {
				const float2 v1 = s_PQ[PQ_index-i];
				const float2 v2 = s_PQ[PQ_index+i];
                l_Px[ZPENCIL_LAST] += c_x[i] * (v1.x + v2.x);
                l_Qx[ZPENCIL_LAST] += c_x[i] * (v1.y + v2.y);
                q_Pxxl += c_xx[i] * (v1.x + v2.x);
                q_Qxxl += c_xx[i] * (v1.y + v2.y);
            }
            Q_LAST(q_Pxx, qidx) = q_Pxxl;
            Q_LAST(q_Qxx, qidx) = q_Qxxl;

			// Compute zz stencils
            float Pzz = 0.f;
            float Qzz = 0.f;
			#pragma unroll
            for(int j=0;j<2*R+1;j++) {
                const float zcoeff = READ_Z_COEFF(c_zz, z_base+j);
                Pzz += zcoeff * l_P[ZPENCIL_CTR_POSTSHIFT-R+j];
                Qzz += zcoeff * l_Q[ZPENCIL_CTR_POSTSHIFT-R+j];
            }

			// Compute y stencil for halo
        	float2 PQyside;
	        PQyside.x = c_y[0] * s_PQ[PQ_index+PQ_halo_diff].x; 
	        PQyside.y = c_y[0] * s_PQ[PQ_index+PQ_halo_diff].y; 
			#pragma unroll
   	    	for(int j=1;j<=R;j++)
			{
				const float2 v1 = s_PQ[PQ_index+PQ_halo_diff-j*COUNT_PER_STRIDE_ROW];
				const float2 v2 = s_PQ[PQ_index+PQ_halo_diff+j*COUNT_PER_STRIDE_ROW];
           		PQyside.x += c_y[j] * (v1.x + v2.x);
				PQyside.y += c_y[j] * (v1.y + v2.y);
			}
        	PQy_buf[PQ_index+PQ_halo_diff] = PQyside;
			
			// Compute y, yy stencils
            l_Py[ZPENCIL_LAST] = c_y[0]*l_P[ZPENCIL_LAST];
            l_Qy[ZPENCIL_LAST] = c_y[0]*l_Q[ZPENCIL_LAST];
            float q_Pyyl = c_yy[0]*l_P[ZPENCIL_LAST];
            float q_Qyyl = c_yy[0]*l_Q[ZPENCIL_LAST];
			#pragma unroll
            for(int j=1;j<=R;j++) {
				const float2 v1 = s_PQ[PQ_index-j*COUNT_PER_STRIDE_ROW];
				const float2 v2 = s_PQ[PQ_index+j*COUNT_PER_STRIDE_ROW];
                l_Py[ZPENCIL_LAST] += c_y[j] * (v1.x + v2.x);
                l_Qy[ZPENCIL_LAST] += c_y[j] * (v1.y + v2.y);
                q_Pyyl += c_yy[j] * (v1.x + v2.x);
                q_Qyyl += c_yy[j] * (v1.y + v2.y);
            }
            Q_LAST(q_Pyy, qidx) = q_Pyyl;
            Q_LAST(q_Qyy, qidx) = q_Qyyl;
            PQy_buf[PQ_index].x = l_Py[ZPENCIL_LAST];
            PQy_buf[PQ_index].y = l_Qy[ZPENCIL_LAST];

			ptx_cudaDMA_barrier_blocking(0,COMPUTE_THREADS_PER_CTA); // __syncthreads()
			dma_ld_pq.start_async_dma(); // fetch next s_PQ

			// Compute xy
            float q_Pxyl = c_x[0] * l_Py[ZPENCIL_LAST];
            float q_Qxyl = c_x[0] * l_Qy[ZPENCIL_LAST];
			#pragma unroll
            for(int j=1;j<=R;j++)
			{
				const float2 v1 = PQy_buf[PQ_index-j];
				const float2 v2 = PQy_buf[PQ_index+j];
                q_Pxyl += c_x[j] * (v1.x + v2.x);
				q_Qxyl += c_x[j] * (v1.y + v2.y);
			}
            Q_LAST(q_Pxy, qidx) = q_Pxyl;
            Q_LAST(q_Qxy, qidx) = q_Qxyl;

            float Pxz = 0.f;
            float Qxz = 0.f;
            float Pyz = 0.f;
            float Qyz = 0.f;
			#pragma unroll
            for(int j=0;j<2*R+1;j++) {
                const float zcoeff = READ_Z_COEFF(c_z, z_base+j);
                Pxz += zcoeff * l_Px[ZPENCIL_CTR_POSTSHIFT-R+j];
                Qxz += zcoeff * l_Qx[ZPENCIL_CTR_POSTSHIFT-R+j];
                Pyz += zcoeff * l_Py[ZPENCIL_CTR_POSTSHIFT-R+j];
                Qyz += zcoeff * l_Qy[ZPENCIL_CTR_POSTSHIFT-R+j];
            }
			
			// Output calculation
			const float  alpha = abde.x;
			const float  beta  = abde.y;
			const float  delta = abde.z;
			const float  epsln = abde.w;

            float sina;
            float cosa;
            __sincosf(alpha, &sina, &cosa);
            float sinb;
            float cosb;
            __sincosf(beta, &sinb, &cosb);

            const float sina2 = sina * sina;
            const float cosa2 = 1.f - sina2;
            const float cosb2 = cosb * cosb;
            const float sinb2 = 1.f - cosb2;
            const float sin2a = 2.f * sina * cosa;
            const float sin2b = 2.f * sinb * cosb;

            const float h1p = sina*cosb2*q_Pxx0 + 
                sina2*sinb2*q_Pyy0 + 
                cosa2*Pzz + 
                sina2*sin2b*q_Pxy0 + 
                sin2a*sinb*Pyz + 
                sin2a*cosb*Pxz;
            const float h2p = q_Pxx0 + q_Pyy0 + Pzz - h1p;

            const float h1q = sina*cosb2*q_Qxx0 + 
                sina2*sinb2*q_Qyy0 + 
                cosa2*Qzz + 
                sina2*sin2b*q_Qxy0 + 
                sin2a*sinb*Qyz + 
                sin2a*cosb*Qxz;
            const float h2q = q_Qxx0 + q_Qyy0 + Qzz - h1q;

            const float vsz2 = vsz2_constant * fabsf(delta - epsln);
            const float vpn2 = vpz2 * (1.f + 2.f*delta);
            const float vpx2 = vpz2 * (1.f + 2.f*epsln);

            REDUCTION_SUB(g_PnQn[out_idx].x, PnQn.x, 2.f*l_P[ZPENCIL_CTR_POSTSHIFT] + vpx2*h2p + vsz2*h1p + vpz2*h1q - vsz2*h1q);
            REDUCTION_SUB(g_PnQn[out_idx].y, PnQn.y, 2.f*l_Q[ZPENCIL_CTR_POSTSHIFT] + vpz2*h1q + vsz2*h2q + vpn2*h2p - vsz2*h2p);
			
			out_idx += slice_stride;
		}
		//------------------------------------------------------
		// "Postamble" for main loop
		//------------------------------------------------------
		dma_ld_pq.wait_for_dma_finish();
		/*
		ZPENCIL_SHIFT(l_P);
		ZPENCIL_SHIFT(l_Px);
		ZPENCIL_SHIFT(l_Py);
		ZPENCIL_SHIFT(l_Q);
		ZPENCIL_SHIFT(l_Qx);
		ZPENCIL_SHIFT(l_Qy);
		const float q_Pxx0 = Q_CURR(q_Pxx, qidx);
		const float q_Qxx0 = Q_CURR(q_Qxx, qidx);
		const float q_Pyy0 = Q_CURR(q_Pyy, qidx);
		const float q_Qyy0 = Q_CURR(q_Qyy, qidx);
		const float q_Pxy0 = Q_CURR(q_Pxy, qidx);
		const float q_Qxy0 = Q_CURR(q_Qxy, qidx);
		ADVANCE_6Qs(q_Pxx, q_Pyy, q_Pxy, q_Qxx, q_Qyy, q_Qxy, qidx);

		// Params
		REDUCTION_LD(const float2 PnQn = ld_PnQn2_ro(out_idx));
		const float4 abde = ld_param_ro(out_idx, abde);
		const float  vpz2 = ld_param_ro(out_idx, vpz2);
		const float  alpha = abde.x;
		const float  beta  = abde.y;
		const float  delta = abde.z;
		const float  epsln = abde.w;

//		dma_ld_pq.wait_for_dma_finish();

		l_P[ZPENCIL_LAST] = s_PQ[PQ_index].x;
		l_Q[ZPENCIL_LAST] = s_PQ[PQ_index].y;

		// Compute x, xx stencils
		l_Px[ZPENCIL_LAST] = c_x[0]*l_P[ZPENCIL_LAST];
		l_Qx[ZPENCIL_LAST] = c_x[0]*l_Q[ZPENCIL_LAST];
		float q_Pxxl = c_xx[0]*l_P[ZPENCIL_LAST];
		float q_Qxxl = c_xx[0]*l_Q[ZPENCIL_LAST];
		#pragma unroll
		for(int i=1;i<=R;i++) {
			const float2 v1 = s_PQ[PQ_index-i];
			const float2 v2 = s_PQ[PQ_index+i];
			l_Px[ZPENCIL_LAST] += c_x[i] * (v1.x + v2.x);
			l_Qx[ZPENCIL_LAST] += c_x[i] * (v1.y + v2.y);
			q_Pxxl += c_xx[i] * (v1.x + v2.x);
			q_Qxxl += c_xx[i] * (v1.y + v2.y);
		}
		Q_LAST(q_Pxx, qidx) = q_Pxxl;
		Q_LAST(q_Qxx, qidx) = q_Qxxl;

		// Compute zz stencils
		float Pzz = 0.f;
		float Qzz = 0.f;
		#pragma unroll
		for(int j=0;j<2*R+1;j++) {
			const float zcoeff = READ_Z_COEFF(c_zz, z_base+j);
			Pzz += zcoeff * l_P[ZPENCIL_CTR_POSTSHIFT-R+j];
			Qzz += zcoeff * l_Q[ZPENCIL_CTR_POSTSHIFT-R+j];
		}

		// Compute y stencil for halo
		float2 PQyside;
		PQyside.x = c_y[0] * s_PQ[PQ_index+PQ_halo_diff].x; 
		PQyside.y = c_y[0] * s_PQ[PQ_index+PQ_halo_diff].y; 
		#pragma unroll
		for(int j=1;j<=R;j++)
		{
			const float2 v1 = s_PQ[PQ_index+PQ_halo_diff-j*COUNT_PER_STRIDE_ROW];
			const float2 v2 = s_PQ[PQ_index+PQ_halo_diff+j*COUNT_PER_STRIDE_ROW];
			PQyside.x += c_y[j] * (v1.x + v2.x);
			PQyside.y += c_y[j] * (v1.y + v2.y);
		}
		PQy_buf[PQ_index+PQ_halo_diff] = PQyside;
		
		// Compute y, yy stencils
		l_Py[ZPENCIL_LAST] = c_y[0]*l_P[ZPENCIL_LAST];
		l_Qy[ZPENCIL_LAST] = c_y[0]*l_Q[ZPENCIL_LAST];
		float q_Pyyl = c_yy[0]*l_P[ZPENCIL_LAST];
		float q_Qyyl = c_yy[0]*l_Q[ZPENCIL_LAST];
		#pragma unroll
		for(int j=1;j<=R;j++) {
			const float2 v1 = s_PQ[PQ_index-j*COUNT_PER_STRIDE_ROW];
			const float2 v2 = s_PQ[PQ_index+j*COUNT_PER_STRIDE_ROW];
			l_Py[ZPENCIL_LAST] += c_y[j] * (v1.x + v2.x);
			l_Qy[ZPENCIL_LAST] += c_y[j] * (v1.y + v2.y);
			q_Pyyl += c_yy[j] * (v1.x + v2.x);
			q_Qyyl += c_yy[j] * (v1.y + v2.y);
		}
		Q_LAST(q_Pyy, qidx) = q_Pyyl;
		Q_LAST(q_Qyy, qidx) = q_Qyyl;
		PQy_buf[PQ_index].x = l_Py[ZPENCIL_LAST];
		PQy_buf[PQ_index].y = l_Qy[ZPENCIL_LAST];

		ptx_cudaDMA_barrier_blocking(0,COMPUTE_THREADS_PER_CTA); // __syncthreads()

		// Compute xy
		float q_Pxyl = c_x[0] * l_Py[ZPENCIL_LAST];
		float q_Qxyl = c_x[0] * l_Qy[ZPENCIL_LAST];
		#pragma unroll
		for(int j=1;j<=R;j++)
		{
			const float2 v1 = PQy_buf[PQ_index-j];
			const float2 v2 = PQy_buf[PQ_index+j];
			q_Pxyl += c_x[j] * (v1.x + v2.x);
			q_Qxyl += c_x[j] * (v1.y + v2.y);
		}
		Q_LAST(q_Pxy, qidx) = q_Pxyl;
		Q_LAST(q_Qxy, qidx) = q_Qxyl;

		float Pxz = 0.f;
		float Qxz = 0.f;
		float Pyz = 0.f;
		float Qyz = 0.f;
		#pragma unroll
		for(int j=0;j<2*R+1;j++) {
			const float zcoeff = READ_Z_COEFF(c_z, z_base+j);
			Pxz += zcoeff * l_Px[ZPENCIL_CTR_POSTSHIFT-R+j];
			Qxz += zcoeff * l_Qx[ZPENCIL_CTR_POSTSHIFT-R+j];
			Pyz += zcoeff * l_Py[ZPENCIL_CTR_POSTSHIFT-R+j];
			Qyz += zcoeff * l_Qy[ZPENCIL_CTR_POSTSHIFT-R+j];
		}
		
		// Output calculation
		float sina;
		float cosa;
		__sincosf(alpha, &sina, &cosa);
		float sinb;
		float cosb;
		__sincosf(beta, &sinb, &cosb);

		const float sina2 = sina * sina;
		const float cosa2 = 1.f - sina2;
		const float cosb2 = cosb * cosb;
		const float sinb2 = 1.f - cosb2;
		const float sin2a = 2.f * sina * cosa;
		const float sin2b = 2.f * sinb * cosb;

		const float h1p = sina*cosb2*q_Pxx0 + 
			sina2*sinb2*q_Pyy0 + 
			cosa2*Pzz + 
			sina2*sin2b*q_Pxy0 + 
			sin2a*sinb*Pyz + 
			sin2a*cosb*Pxz;
		const float h2p = q_Pxx0 + q_Pyy0 + Pzz - h1p;

		const float h1q = sina*cosb2*q_Qxx0 + 
			sina2*sinb2*q_Qyy0 + 
			cosa2*Qzz + 
			sina2*sin2b*q_Qxy0 + 
			sin2a*sinb*Qyz + 
			sin2a*cosb*Qxz;
		const float h2q = q_Qxx0 + q_Qyy0 + Qzz - h1q;

		const float vsz2 = vsz2_constant * fabsf(delta - epsln);
		const float vpn2 = vpz2 * (1.f + 2.f*delta);
		const float vpx2 = vpz2 * (1.f + 2.f*epsln);

		REDUCTION_SUB(g_PnQn[out_idx].x, PnQn.x, 2.f*l_P[ZPENCIL_CTR_POSTSHIFT] + vpx2*h2p + vsz2*h1p + vpz2*h1q - vsz2*h1q);
		REDUCTION_SUB(g_PnQn[out_idx].y, PnQn.y, 2.f*l_Q[ZPENCIL_CTR_POSTSHIFT] + vpz2*h1q + vsz2*h2q + vpn2*h2p - vsz2*h2p);
		*/
	}
	else if(dma_ld_pq.owns_this_thread())
	{
		int in_idx = gid - R_x_row_stride - R 
						 - R*slice_stride;
#ifdef DYNAMIC
		for(int i=0; i<(2*R+nz); i++)
#else
                #pragma unroll 1
                for(int i=0; i<(2*R+RTM_ELMTS); i++)
#endif
		{
			dma_ld_pq.execute_dma(&g_PQ[in_idx],s_PQ);
			in_idx += slice_stride;
		}
	}
}


