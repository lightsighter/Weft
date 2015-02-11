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

#ifndef __WEFT_H__
#define __WEFT_H__

#include <cstdio>
#include <cassert>
#include <pthread.h>
#include <deque>
#include <vector>
#include <string>

#define PTHREAD_SAFE_CALL(cmd)        \
  {                                   \
    int ret = (cmd);                  \
    if (ret != 0) {                   \
      fprintf(stderr,"PTHREAD error: %s = %d (%s)\n", #cmd, ret, strerror(ret)); \
      assert(false);                  \
    }                                 \
  }

#define WARP_SIZE   32

enum {
  WEFT_SUCCESS,
  WEFT_ERROR_NO_FILE_NAME,
  WEFT_ERROR_FILE_OPEN,
  WEFT_ERROR_MULTIPLE_KERNELS,
  WEFT_ERROR_THREAD_COUNT_MISMATCH,
  WEFT_ERROR_NO_THREAD_COUNT,
  WEFT_ERROR_ARRIVAL_MISMATCH,
  WEFT_ERROR_TOO_MANY_PARTICIPANTS,
  WEFT_ERROR_ALL_ARRIVALS,
  WEFT_ERROR_DEADLOCK,
  WEFT_ERROR_GRAPH_VALIDATION,
};

class Weft;
class Thread;
class Program;
class Address;
class SharedMemory;
class BarrierInstance;
class BarrierDependenceGraph;

class WeftTask {
public:
  virtual ~WeftTask(void) { }
  virtual void execute(void) = 0;
};

class EmulateThread : public WeftTask {
public:
  EmulateThread(Thread *thread);
  EmulateThread(const EmulateThread &rhs) : thread(NULL) { assert(false); }
  virtual ~EmulateThread(void) { }
public:
  EmulateThread& operator=(const EmulateThread &rhs) { assert(false); return *this; }
public:
  virtual void execute(void);
public:
  Thread *const thread;
};

class EmulateWarp : public WeftTask {
public:
  EmulateWarp(Program *p, Thread **start);
  EmulateWarp(const EmulateWarp &rhs) : program(NULL), threads(NULL) { assert(false); }
  virtual ~EmulateWarp(void) { }
public:
  EmulateWarp& operator=(const EmulateWarp &rhs) { assert(false); return *this; }
public:
  virtual void execute(void);
public:
  Program *const program;
  Thread **const threads;
};

class ValidationTask : public WeftTask {
public:
  ValidationTask(BarrierDependenceGraph *graph, int name, int generation);
  ValidationTask(const ValidationTask &rhs) : graph(NULL), 
    name(0), generation(0) { assert(false); }
  virtual ~ValidationTask(void) { }
public:
  ValidationTask& operator=(const ValidationTask &rhs) { assert(false); return *this; }
public:
  virtual void execute(void);
public:
  BarrierDependenceGraph *const graph;
  const int name;
  const int generation;
};

class InitializationTask : public WeftTask {
public:
  InitializationTask(Thread *thread, int total, int max_num_barriers);
  InitializationTask(const InitializationTask &rhs) 
    : thread(NULL), total_threads(0), max_num_barriers(0) { assert(false); }
  virtual ~InitializationTask(void) { }
public:
  InitializationTask& operator=(const InitializationTask &rhs)
    { assert(false); return *this; }
public:
  virtual void execute(void);
public:
  Thread *const thread;
  const int total_threads;
  const int max_num_barriers;
};

class ReachabilityTask : public WeftTask {
public:
  ReachabilityTask(BarrierInstance *instance, Weft *weft, bool forward);
  ReachabilityTask(const ReachabilityTask &rhs) : instance(NULL),
    weft(NULL), forward(true) { assert(false); }
  virtual ~ReachabilityTask(void) { }
public:
  ReachabilityTask& operator=(const ReachabilityTask &rhs)
    { assert(false); return *this; }
public:
  virtual void execute(void);
public:
  BarrierInstance *const instance;
  Weft *const weft;
  const bool forward;
};

class TransitiveTask : public WeftTask {
public:
  TransitiveTask(BarrierInstance *instance, Weft *weft, bool forward);
  TransitiveTask(const TransitiveTask &rhs) : instance(NULL),
    weft(NULL), forward(true) { assert(false); }
  virtual ~TransitiveTask(void) { }
public:
  TransitiveTask& operator=(const TransitiveTask &rhs)
    { assert(false); return *this; }
public:
  virtual void execute(void);
public:
  BarrierInstance *const instance;
  Weft *const weft;
  const bool forward;
};

class UpdateThreadTask : public WeftTask {
public:
  UpdateThreadTask(Thread *thread);
  UpdateThreadTask(const UpdateThreadTask &rhs) : thread(NULL) { assert(false); }
  virtual ~UpdateThreadTask(void) { }
public:
  UpdateThreadTask& operator=(const UpdateThreadTask &rhs) 
    { assert(false); return *this; }  
public:
  virtual void execute(void);
public:
  Thread *const thread;
};

class RaceCheckTask : public WeftTask {
public:
  RaceCheckTask(Address *address);
  RaceCheckTask(const RaceCheckTask &rhs) : address(NULL) { assert(false); }
  virtual ~RaceCheckTask(void) { }
public:
  RaceCheckTask& operator=(const RaceCheckTask &rhs)
    { assert(false); return *this; }
public:
  virtual void execute(void);
public:
  Address *const address;
};

class DumpThreadTask : public WeftTask {
public:
  DumpThreadTask(Thread *thread);
  DumpThreadTask(const DumpThreadTask &rhs) : thread(NULL) { assert(false); }
  virtual ~DumpThreadTask(void) { }
public:
  DumpThreadTask& operator=(const DumpThreadTask &rhs)
    { assert(false); return *this; }
public:
  virtual void execute(void);
public:
  Thread *const thread;
};

class Weft {
public:
  Weft(int argc, char **argv);
  ~Weft(void);
public:
  void verify(void);
  void report_error(int error_code, const char *message);
  inline bool report_warnings(void) const { return warnings; }
  inline int thread_count(void) const { return max_num_threads; }
  inline int barrier_upper_bound(void) const { return max_num_barriers; }
  inline bool print_verbose(void) const { return verbose; }
  inline bool print_detail(void) const { return detailed; }
  inline bool assume_warp_synchronous(void) const { return warp_synchronous; }
protected:
  void parse_inputs(int argc, char **argv);
  bool parse_triple(const std::string &input, int *array,
                    const char *flag, const char *error_str);
  void report_usage(int error, const char *error_str);
  void parse_ptx(void);
  void emulate_threads(void); 
  void construct_dependence_graph(void);
  void compute_happens_relationships(void);
  void check_for_race_conditions(void);
  void print_statistics(void);
  int count_dynamic_instructions(void);
  int count_weft_statements(void);
public:
  void fill_block_dim(int *array);
  void fill_block_id(int *array);
  void fill_grid_dim(int *array);
  void get_file_prefix(char *buffer, size_t count);
protected:
  void start_threadpool(void);
  void stop_threadpool(void);
  void initialize_count(unsigned count);
  void wait_until_done(void);
public:
  void enqueue_task(WeftTask *task);
  WeftTask* dequeue_task(void);
  void complete_task(WeftTask *task);
public:
  static void* worker_loop(void *arg);
  static unsigned long long get_current_time_in_micros(void);
  static size_t get_memory_usage(void);
protected:
  void start_instrumentation(int stage);
  void stop_instrumentation(int stage);
  void report_instrumentation(void);
protected:
  const char *file_name;
  int block_dim[3]; // x, y, z
  int block_id[3]; // x, y, z
  int grid_dim[3]; // x, y, z
  int max_num_threads;
  int thread_pool_size;
  int max_num_barriers;
  bool verbose;
  bool detailed;
  bool instrument;
  bool warnings;
  bool warp_synchronous;
  bool print_files;
protected:
  Program *program;
  std::vector<Thread*> threads;
  SharedMemory *shared_memory;
protected:
  BarrierDependenceGraph *graph;
protected:
  pthread_t *worker_threads;
  bool threadpool_finished;
protected:
  pthread_mutex_t count_lock;
  pthread_cond_t count_cond;
  unsigned int pending_count;
protected:
  pthread_mutex_t queue_lock;
  pthread_cond_t queue_cond;
  std::deque<WeftTask*> queue;
protected:
  // Instrumentation
  unsigned long long timing[5];
  size_t memory_usage[5];
};

#endif // __WEFT_H__
