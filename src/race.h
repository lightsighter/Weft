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

#include <map>
#include <set>
#include <deque>
#include <vector>
#include <cassert>
#include <pthread.h>

class Weft;
class WeftAccess;
class WeftBarrier;
class SharedMemory;

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
  bool has_happens(int thread, int line_number);
protected:
  bool initialized;
  std::vector<WeftBarrier*> latest_before;
  std::vector<WeftBarrier*> earliest_after;
  std::vector<int> happens_before;
  std::vector<int> happens_after;
};

class Address {
public:
  Address(const int addr, SharedMemory *memory);
  Address(const Address &rhs) : address(0), memory(NULL) { assert(false); }
  ~Address(void);
public:
  Address& operator=(const Address &rhs) { assert(false); return *this; }
public:
  void add_access(WeftAccess *access);
  void perform_race_tests(void);
  int report_races(void);
  size_t count_race_tests(void);
protected:
  void record_race(WeftAccess *one, WeftAccess *two);
public:
  const int address;
  SharedMemory *const memory;
protected:
  pthread_mutex_t address_lock;
  std::vector<WeftAccess*> accesses;
protected:
  int total_races;
  std::set<std::pair<int,int> > ptx_races;
};

class SharedMemory {
public:
  SharedMemory(Weft *weft);
  SharedMemory(const SharedMemory &rhs) : weft(NULL) { assert(false); }
  ~SharedMemory(void);
public:
  SharedMemory& operator=(const SharedMemory &rhs) 
    { assert(false); return *this; }
public:
  void update_accesses(WeftAccess *access);
  int count_addresses(void) const;
  void enqueue_race_checks(void);
  void check_for_races(void);
  size_t count_race_tests(void);
public:
  Weft *const weft;
protected:
  pthread_mutex_t memory_lock;
  std::map<int/*address*/,Address*> addresses;
};

#endif // __RACE_H__
