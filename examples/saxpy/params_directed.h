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

#define SAXPY_KERNEL saxpy_cudaDMA_doublebuffer
#define CTA_COUNT 14
#define COMPUTE_THREADS_PER_CTA 32 * 8
#ifndef NUM_ITERS
#define NUM_ITERS 2048
#endif
#define DMA_THREADS_PER_LD 32 * 1
#define BYTES_PER_DMA_THREAD 32
#define DMA_SZ 4 * COMPUTE_THREADS_PER_CTA
