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

__global__ void
__launch_bounds__(64,1)
deadlock_test(void)
{
  int wid = threadIdx.x >> 5;
  if (wid == 0) {
    asm volatile("bar.sync 0, 64;" : : : "memory");
    asm volatile("bar.arrive 1, 64;" : : : "memory");
  } else {
    asm volatile("bar.sync 1, 64;" : : : "memory");
    asm volatile("bar.arrive 0, 64;" : : : "memory");
  }
}
