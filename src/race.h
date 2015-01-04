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

#ifndef __RACE_H__
#define __RACE_H__

#include <vector>
#include <cassert>

class WeftBarrier;

class Happens {
public:
  Happens(int total_threads);
  Happens(const Happens &rhs) { assert(false); }
  ~Happens(void) { }
public:
  Happens& operator=(const Happens &rhs) { assert(false); return *this; }
public:
  void update_barriers_before(const std::vector<WeftBarrier*> &before);
  void update_barriers_after(const std::vector<WeftBarrier*> &after);
public:
  void update_happens_relationships(void);
protected:
  bool initialized;
  std::vector<WeftBarrier*> latest_before;
  std::vector<WeftBarrier*> earliest_after;
  std::vector<int> happens_before;
  std::vector<int> happens_after;
};

#endif // __RACE_H__
