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

#include "weft.h"
#include "race.h"
#include "graph.h"
#include "program.h"
#include "instruction.h"

#include <string>

#include <cstdio>
#include <cassert>
#include <cstring>
#include <cstdlib>

#include <sys/time.h>
#include <sys/resource.h>

#ifdef __MACH__
#include "mach/clock.h"
#include "mach/mach.h"
#endif

Weft::Weft(int argc, char **argv)
  : file_name(NULL), max_num_threads(-1), 
    thread_pool_size(1), max_num_barriers(1),
    verbose(false), detailed(false), instrument(false), 
    warnings(false), warp_synchronous(false), print_files(false),
    program(NULL), shared_memory(NULL), graph(NULL),
    worker_threads(NULL), pending_count(0)
{
  for (int i = 0; i < 3; i++)
    block_dim[i] = 1;
  for (int i = 0; i < 3; i++)
    block_id[i] = 0;
  for (int i = 0; i < 3; i++)
    grid_dim[i] = 1;
  parse_inputs(argc, argv);  
  start_threadpool();
}

Weft::~Weft(void)
{
  stop_threadpool();
  if (program != NULL)
  {
    delete program;
    program = NULL;
  }
  if (shared_memory != NULL)
  {
    delete shared_memory;
    shared_memory = NULL;
  }
  if (graph != NULL)
  {
    delete graph;
    graph = NULL;
  }
  for (std::vector<Thread*>::iterator it = threads.begin();
        it != threads.end(); it++)
  {
    delete (*it);
  }
  threads.clear();
  if (instrument)
    report_instrumentation();
}

void Weft::verify(void)
{
  parse_ptx();
  emulate_threads();
  construct_dependence_graph();
  compute_happens_relationships();
  check_for_race_conditions();
  print_statistics();
}

void Weft::report_error(int error_code, const char *message)
{
  assert(error_code != WEFT_SUCCESS);
  fprintf(stderr,"WEFT ERROR %d: %s!\n", error_code, message);
  fprintf(stderr,"WEFT WILL NOW EXIT...\n");
  fflush(stderr);
  stop_threadpool();
  exit(error_code);
}

void Weft::parse_inputs(int argc, char **argv)
{
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i],"-b"))
    {
      std::string block(argv[++i]);
      parse_triple(block, block_id, "-b", "CTA ID");
      continue;
    }
    if (!strcmp(argv[i],"-d"))
    {
      detailed = true;
      continue;
    }
    if (!strcmp(argv[i],"-f"))
    {
      file_name = argv[++i];
      continue;
    }
    if (!strcmp(argv[i],"-g"))
    {
      std::string grid(argv[++i]);
      parse_triple(grid, grid_dim, "-g", "Grid Size");
      continue;
    }
    if (!strcmp(argv[i],"-i"))
    {
      instrument = true;
      continue;
    }
    if (!strcmp(argv[i],"-p"))
    {
      print_files = true;
      continue;
    }
    if (!strcmp(argv[i],"-n"))
    {
      std::string threads(argv[++i]);
      // If we succeeded, compute the max number of threads
      if (parse_triple(threads, block_dim, "-n", "CTA size"))
        max_num_threads = block_dim[0] * block_dim[1] * block_dim[2];
      continue;
    }
    if (!strcmp(argv[i],"-s"))
    {
      warp_synchronous = true;
      continue;
    }
    if (!strcmp(argv[i],"-t"))
    {
      thread_pool_size = atoi(argv[++i]);
      if (thread_pool_size < 1)
        thread_pool_size = 1;
      continue;
    }
    if (!strcmp(argv[i],"-v"))
    {
      verbose = true;
      continue;
    }
    if (!strcmp(argv[i],"-w"))
    {
      warnings = true;
      continue;
    }
    // If it has a ptx ending then guess it is the file name
    std::string file(argv[i]);
    if (file.find(".ptx") != std::string::npos)
    {
      file_name = argv[i];
      continue;
    }
    fprintf(stderr,"WEFT WARNING: skipping argument %s\n", argv[i]);
  }
  if (file_name == NULL)
    report_usage(WEFT_ERROR_NO_FILE_NAME, "No file name specified");
  if (verbose)
  {
    fprintf(stdout,"INITIAL WEFT SETTINGS:\n");
    fprintf(stdout,"  File Name: %s\n", file_name);
    fprintf(stdout,"  CTA dimensions: (%d,%d,%d)\n", 
                      block_dim[0], block_dim[1], block_dim[2]);
    fprintf(stdout,"  Block ID: (%d,%d,%d)\n",
                      block_id[0], block_id[1], block_id[2]);
    fprintf(stdout,"  Grid dimensions: (%d,%d,%d)\n",
                      grid_dim[0], grid_dim[1], grid_dim[2]);
    fprintf(stdout,"  Thread Pool Size: %d\n", thread_pool_size);
    fprintf(stdout,"  Verbose: %s\n", (verbose ? "yes" : "no"));
    fprintf(stdout,"  Detailed: %s\n", (detailed ? "yes" : "no"));
    fprintf(stdout,"  Instrument: %s\n", (instrument ? "yes" : "no"));
    fprintf(stdout,"  Report Warnings: %s\n", (warnings ? "yes" : "no"));
    fprintf(stdout,"  Warp-Synchronous Execution: %s\n", (warnings ? "yes" : "no"));
    fprintf(stdout,"  Dump Weft thread files: %s\n", (print_files ? "yes" : "no"));
  }
}

bool Weft::parse_triple(const std::string &input, int *array,
                        const char *flag, const char *error_str)
{
  bool success = true;
  if (input.find("x") != std::string::npos)
  {
    // Try parsing this block configuration
    std::vector<std::string> values;
    split(values, input.c_str(), 'x');
    if (!values.empty() && (values.size() <= 3))
    {
      // Try parsing each of the arguments   
      for (unsigned i = 0; i < values.size(); i++)
      {
        int count = atoi(values[i].c_str());
        if (count < 1)
        {
          fprintf(stderr,"WEFT WARNING: Failed to parse dimension %d "
                         "of %s: \"%s %s\"!\n",
                         i, error_str, flag, input.c_str());
          success = false;
          break;
        }
        array[i] = count;
      }
    }
    else
    {
      fprintf(stderr,"WEFT WARNING: Failed to parse %s with %ld"
               "dimensions from input: \"%s %s\"!\n", 
               error_str, values.size(), flag, input.c_str());
      success = false;
    }
  }
  else
  {
    int count = atoi(input.c_str());
    if (count >= 1)
      array[0] = count;
    else
    {
      success = false;
      fprintf(stderr,"WEFT WARNING: Ignoring invalid input for %s "
                     "\"%s %s\"!\n", error_str, flag, input.c_str());
    }
  }
  return success;
}

void Weft::report_usage(int error, const char *error_str)
{
  fprintf(stderr,"WEFT ERROR %d: %s!\nWEFT WILL NOW EXIT...\n", 
          error, error_str);
  fprintf(stderr,"Usage: Weft [args]\n");
  fprintf(stderr,"  -b: specify the CTA id to simulate (default 0x0x0)\n");
  fprintf(stderr,"      can be an integer or an x-separated tuple e.g. 0x0x1 or 1x2\n");
  fprintf(stderr,"  -d: print detailed information for error reporting\n");
  fprintf(stderr,"      this includes line numbers for blocked threads under deadlock and\n");
  fprintf(stderr,"      and per-thread and per-address information for races\n");
  fprintf(stderr,"  -f: specify the input file\n");
  fprintf(stderr,"  -g: specify the grid dimensions for the kernel being simulated\n");
  fprintf(stderr,"      can be an integer or an x-separated tuple e.g. 32x32x2 or 32x1\n");
  fprintf(stderr,"      Weft will still only simulate a single CTA specified by '-b'\n");
  fprintf(stderr,"  -i: instrument execution\n");
  fprintf(stderr,"  -p: print individual Weft thread files (one file per thread!)\n");
  fprintf(stderr,"  -n: number of threads per CTA\n");
  fprintf(stderr,"      can be an integer or an x-separated tuple e.g. 64x2 or 32x8x1\n");
  fprintf(stderr,"  -s: assume warp-synchronous execution\n");
  fprintf(stderr,"  -t: thread pool size\n");
  fprintf(stderr,"  -v: print verbose output\n");
  fprintf(stderr,"  -w: report emulation warnings (this will generate considerable output)\n");
  exit(error);
}

void Weft::parse_ptx(void)
{
  assert(file_name != NULL);
  if (verbose)
    fprintf(stdout,"WEFT INFO: Parsing file %s...\n", file_name);
  if (instrument)
    start_instrumentation(0/*stage*/);
  assert(program == NULL);
  program = new Program(this);
  bool need_update = (max_num_threads == -1);
  program->parse_ptx_file(file_name, max_num_threads);
  // If we didn't get a block size, make it Nx1x1
  if (need_update)
    block_dim[0] = max_num_threads;
  if (max_num_threads <= 0)
  {
    char buffer[1024];
    snprintf(buffer, 1023," Failed to find max number of threads "
                   "in file %s and the value was not set on the command "
                   "line using the '-n' flag", file_name);
    report_error(WEFT_ERROR_NO_THREAD_COUNT, buffer);
  }
  // Check for shuffles, if we have shuffles then make sure
  // that we have enabled warp-synchronous execution
  if (!warp_synchronous && program->has_shuffles())
  {
    fprintf(stdout,"WEFT WARNING: Program has shuffle instructions "
                   "but warp-synchronous execution was not assumed!\n"
                   "Enabling warp-synchronous assumption...\n");
    warp_synchronous = true;
  }
  if (instrument)
    stop_instrumentation(0/*stage*/);
  if (verbose)
    program->report_statistics();
}

void Weft::emulate_threads(void)
{
  if (verbose)
    fprintf(stdout,"WEFT INFO: Emulating %d GPU threads "
                   "with %d CPU threads...\n",
                   max_num_threads, thread_pool_size);
  if (instrument)
    start_instrumentation(1/*stage*/);
  assert(shared_memory == NULL);
  shared_memory = new SharedMemory(this);
  assert(max_num_threads > 0);
  assert(max_num_threads == (block_dim[0]*block_dim[1]*block_dim[2]));
  threads.resize(max_num_threads, NULL);
  // If we are doing warp synchronous execution we 
  // execute all the threads in a warp together
  if (warp_synchronous) 
  {
    assert((max_num_threads % WARP_SIZE) == 0);
    initialize_count(max_num_threads/WARP_SIZE);
    int tid = 0;
    for (int z = 0; z < block_dim[2]; z++)
    {
      for (int y = 0; y < block_dim[1]; y++)
      {
        for (int x = 0; x < block_dim[0]; x++)
        {
          threads[tid] = new Thread(tid, x, y, z, program, shared_memory);
          // Increment first
          tid++;
          // Only kick off a warp once we've generated all the threads
          if ((tid % WARP_SIZE) == 0)
          {
            assert((tid-WARP_SIZE) >= 0);
            EmulateWarp *task = 
              new EmulateWarp(program, &(threads[tid-WARP_SIZE]));
            enqueue_task(task);
          }
        }
      }
    }
  }
  else
  {
    initialize_count(max_num_threads);
    int tid = 0;
    for (int z = 0; z < block_dim[2]; z++)
    {
      for (int y = 0; y < block_dim[1]; y++)
      {
        for (int x = 0; x < block_dim[0]; x++)
        {
          threads[tid] = new Thread(tid, x, y, z, program, shared_memory); 
          EmulateThread *task = new EmulateThread(threads[tid]);
          enqueue_task(task);
          tid++;
        }
      }
    }
  }
  wait_until_done();
  // Get the maximum barrier ID from all threads
  for (int i = 0; i < max_num_threads; i++)
  {
    int local_max = threads[i]->get_max_barrier_name();
    if ((local_max+1) > max_num_barriers)
      max_num_barriers = (local_max+1);
  }
  if (verbose)
  {
    fprintf(stdout,"WEFT INFO: Emulation found %d named barriers.\n",
                    max_num_barriers);
    program->report_statistics(threads);
  }

  if (instrument)
    stop_instrumentation(1/*stage*/);

  // If we want to dump thread-specific files, do that now
  // Note that we don't include this in the timing
  if (print_files)
  {
    initialize_count(max_num_threads);
    for (std::vector<Thread*>::const_iterator it = threads.begin();
          it != threads.end(); it++)
    {
      DumpThreadTask *dump_task = new DumpThreadTask(*it); 
      enqueue_task(dump_task);
    }
    wait_until_done();
  }
}

void Weft::construct_dependence_graph(void)
{
  if (verbose)
    fprintf(stdout,"WEFT INFO: Constructing barrier dependence graph...\n");
  if (instrument)
    start_instrumentation(2/*stage*/);

  assert(graph == NULL);
  graph = new BarrierDependenceGraph(this);
  graph->construct_graph(threads);

  // Validate the graph 
  int total_validation_tasks = graph->count_validation_tasks();
  if (verbose)
    fprintf(stdout,"WEFT INFO: Performing %d graph validation checks...\n",
                              total_validation_tasks);
  if (total_validation_tasks > 0)
  {
    initialize_count(total_validation_tasks);
    graph->enqueue_validation_tasks();
    wait_until_done();
    graph->check_for_validation_errors();
  }

  if (instrument)
    stop_instrumentation(2/*stage*/);
}

void Weft::compute_happens_relationships(void)
{
  if (verbose)
    fprintf(stdout,"WEFT INFO: Computing happens-before/after "
                   "relationships...\n");
  if (instrument)
    start_instrumentation(3/*stage*/);

  // First initialize all the data structures
  initialize_count(threads.size());
  for (std::vector<Thread*>::const_iterator it = threads.begin();
        it != threads.end(); it++)
    enqueue_task(new InitializationTask(*it, threads.size(), max_num_barriers));
  wait_until_done();

  // Compute barrier reachability
  // There are twice as many tasks as barriers
  int total_barriers = graph->count_total_barriers();
  initialize_count(2*total_barriers);
  graph->enqueue_reachability_tasks();
  wait_until_done();

  // Compute latest/earliest happens-before/after tasks
  // There are twice as many tasks as barriers
  initialize_count(2*total_barriers);
  graph->enqueue_transitive_happens_tasks();
  wait_until_done();

  // Finally update all the happens relationships
  initialize_count(threads.size());
  for (std::vector<Thread*>::const_iterator it = threads.begin();
        it != threads.end(); it++)
    enqueue_task(new UpdateThreadTask(*it));
  wait_until_done();

  if (instrument)
    stop_instrumentation(3/*stage*/);
}

void Weft::check_for_race_conditions(void)
{
  if (verbose)
    fprintf(stdout,"WEFT INFO: Checking for race conditions...\n");
  if (instrument)
    start_instrumentation(4/*stage*/);

  initialize_count(shared_memory->count_addresses());
  shared_memory->enqueue_race_checks();
  wait_until_done();
  shared_memory->check_for_races();

  if (instrument)
    stop_instrumentation(4/*stage*/);
}

void Weft::print_statistics(void)
{
  fprintf(stdout,"WEFT STATISTICS for %s\n", file_name);
  fprintf(stdout,"  CTA Thread Count:          %15d\n", max_num_threads);
  fprintf(stdout,"  Shared Memory Locations:   %15d\n", 
                                    shared_memory->count_addresses());
  fprintf(stdout,"  Physical Named Barriers;   %15d\n", max_num_barriers);
  fprintf(stdout,"  Dynamic Barrier Instances: %15d\n", 
                                    graph->count_total_barriers());
  fprintf(stdout,"  Static Instructions:       %15d\n", 
                                    program->count_instructions());
  fprintf(stdout,"  Dynamic Instructions:      %15d\n",
                                    count_dynamic_instructions());
  fprintf(stdout,"  Weft Statements:           %15d\n",
                                    count_weft_statements());   
  fprintf(stdout,"  Total Race Tests:          %15ld\n",
                                    shared_memory->count_race_tests());
}

int Weft::count_dynamic_instructions(void)
{
  int result = 0;
  for (std::vector<Thread*>::const_iterator it = threads.begin();
        it != threads.end(); it++)
  {
    result += (*it)->count_dynamic_instructions();
  }
  return result;
}

int Weft::count_weft_statements(void)
{
  int result = 0;
  for (std::vector<Thread*>::const_iterator it = threads.begin();
        it != threads.end(); it++)
  {
    result += (*it)->count_weft_statements();
  }
  return result;
}

void Weft::fill_block_dim(int *array)
{
  for (int i = 0; i < 3; i++)
    array[i] = block_dim[i];
}

void Weft::fill_block_id(int *array)
{
  for (int i = 0; i < 3; i++)
    array[i] = block_id[i];
}

void Weft::fill_grid_dim(int *array)
{
  for (int i = 0; i < 3; i++)
    array[i] = grid_dim[i];
}

void Weft::get_file_prefix(char *buffer, size_t count)
{
  std::string full_name(file_name);
  assert(full_name.find(".ptx") != std::string::npos);
  std::string base = full_name.substr(0, full_name.find(".ptx"));
  strncpy(buffer, base.c_str(), count);
}

void Weft::start_threadpool(void)
{
  assert(thread_pool_size > 0);
  PTHREAD_SAFE_CALL( pthread_mutex_init(&count_lock, NULL) );
  PTHREAD_SAFE_CALL( pthread_cond_init(&count_cond, NULL) );
  PTHREAD_SAFE_CALL( pthread_mutex_init(&queue_lock, NULL) );
  PTHREAD_SAFE_CALL( pthread_cond_init(&queue_cond, NULL) );
  assert(worker_threads == NULL);
  worker_threads = (pthread_t*)malloc(thread_pool_size * sizeof(pthread_t));
  threadpool_finished = false;
  for (int i = 0; i < thread_pool_size; i++)
  {
    PTHREAD_SAFE_CALL( pthread_create(worker_threads+i, NULL, 
                                      Weft::worker_loop, this) );
  }
}

void Weft::stop_threadpool(void)
{
  // Wake up all the worker threads so that they exit
  PTHREAD_SAFE_CALL( pthread_mutex_lock(&queue_lock) );
  threadpool_finished = true;
  PTHREAD_SAFE_CALL( pthread_cond_broadcast(&queue_cond) );
  PTHREAD_SAFE_CALL( pthread_mutex_unlock(&queue_lock) );
  for (int i = 0; i < thread_pool_size; i++)
  {
    PTHREAD_SAFE_CALL( pthread_join(worker_threads[i], NULL) ) ;
  }
  free(worker_threads);
  worker_threads = NULL;
  PTHREAD_SAFE_CALL( pthread_mutex_destroy(&count_lock) );
  PTHREAD_SAFE_CALL( pthread_cond_destroy(&count_cond) );
  PTHREAD_SAFE_CALL( pthread_mutex_destroy(&queue_lock) );
  PTHREAD_SAFE_CALL( pthread_cond_destroy(&queue_cond) );
}

void Weft::initialize_count(unsigned count)
{
  PTHREAD_SAFE_CALL( pthread_mutex_lock(&count_lock) ); 
  assert(pending_count == 0);
  pending_count = count;
  PTHREAD_SAFE_CALL( pthread_mutex_unlock(&count_lock) );
}

void Weft::wait_until_done(void)
{
  PTHREAD_SAFE_CALL( pthread_mutex_lock(&count_lock) );
  if (pending_count > 0)
  {
    PTHREAD_SAFE_CALL( pthread_cond_wait(&count_cond, &count_lock) );
  }
  PTHREAD_SAFE_CALL( pthread_mutex_unlock(&count_lock) );
}

void Weft::enqueue_task(WeftTask *task)
{
  PTHREAD_SAFE_CALL( pthread_mutex_lock(&queue_lock) );
  queue.push_back(task); 
  PTHREAD_SAFE_CALL( pthread_cond_signal(&queue_cond) );
  PTHREAD_SAFE_CALL( pthread_mutex_unlock(&queue_lock) );
}

WeftTask* Weft::dequeue_task(void)
{
  WeftTask *result = NULL;
  bool done = false;
  while (!done)
  {
    PTHREAD_SAFE_CALL( pthread_mutex_lock(&queue_lock) );
    if (queue.empty()) 
    {
      if (!threadpool_finished)
      {
        PTHREAD_SAFE_CALL( pthread_cond_wait(&queue_cond, &queue_lock) );
      }
      else
        done = true;
    }
    else
    {
      result = queue.front();
      queue.pop_front();
      done = true;
    }
    PTHREAD_SAFE_CALL( pthread_mutex_unlock(&queue_lock) );
  }
  return result;
}

void Weft::complete_task(WeftTask *task)
{
  PTHREAD_SAFE_CALL( pthread_mutex_lock(&count_lock) );
  assert(pending_count > 0);
  pending_count--;
  if (pending_count == 0)
    PTHREAD_SAFE_CALL( pthread_cond_signal(&count_cond) );
  PTHREAD_SAFE_CALL( pthread_mutex_unlock(&count_lock) );
  // Clean up the task
  delete task;
}

/*static*/
void* Weft::worker_loop(void *arg)
{
  Weft *weft = (Weft*)arg;
  while (true)
  {
    WeftTask *task = weft->dequeue_task();
    // If we ever get a NULL task then we are done
    if (task == NULL)
      break;
    task->execute();
    weft->complete_task(task);
  }
  return NULL;
}

/*static*/
unsigned long long Weft::get_current_time_in_micros(void)
{
#ifdef __MACH__
  mach_timespec_t spec;
  clock_serv_t cclock;
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time(cclock, &spec);
  mach_port_deallocate(mach_host_self(), cclock);
#else
  struct timespec spec;
  clock_gettime(CLOCK_MONOTONIC, &spec);
#endif
  unsigned long long result = (((unsigned long long)spec.tv_sec) * 1000000) +
                              (((unsigned long long)spec.tv_nsec) / 1000);
  return result;
}

/*static*/
size_t Weft::get_memory_usage(void)
{
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_maxrss;
}

void Weft::start_instrumentation(int stage)
{
  timing[stage] = get_current_time_in_micros();
}

void Weft::stop_instrumentation(int stage)
{
  unsigned long long stop = get_current_time_in_micros();
  unsigned long long start = timing[stage];
  timing[stage] = stop - start;
  memory_usage[stage] = get_memory_usage();
}

void Weft::report_instrumentation(void)
{
  const char *stage_names[5] = { "Parse PTX", "Emulate Threads",
                                 "Construct Barrier Dependence Graph",
                                 "Compute Happens-Before/After Relationships",
                                 "Check for Race Conditions" };
  fprintf(stdout,"WEFT INSTRUMENTATION\n");
  unsigned long long total_time = 0;
  size_t total_memory = 0;
  for (int i = 0; i < 5; i++)
  {
    double time = double(timing[i]) * 1e-3;
    size_t memory = memory_usage[i] - total_memory;
    fprintf(stdout,"  %50s: %10.3lf ms %12ld MB\n",
            stage_names[i], time, memory / 1024);
    total_time += timing[i];
    total_memory += memory;
  }
  fprintf(stdout,"  %50s: %10.3lf ms %12ld MB\n",
          "Total", double(total_time) * 1e-3, total_memory / 1024);
}

int main(int argc, char **argv)
{
  Weft weft(argc, argv);
  weft.verify();
  fflush(stderr);
  fflush(stdout);
  return 0;
}

