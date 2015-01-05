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

// These are computed from the above parameters
#define DMA_THREADS_PER_CTA ( (SAXPY_KERNEL==saxpy_cudaDMA_doublebuffer) ? 4 : 2 ) * DMA_THREADS_PER_LD
#define THREADS_PER_CTA \
  (SAXPY_KERNEL==saxpy_cudaDMA_doublebuffer) ? (COMPUTE_THREADS_PER_CTA+DMA_THREADS_PER_CTA) : \
  (SAXPY_KERNEL==saxpy_cudaDMA) ? (COMPUTE_THREADS_PER_CTA+DMA_THREADS_PER_CTA) : \
  COMPUTE_THREADS_PER_CTA
