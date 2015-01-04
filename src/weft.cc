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
#include "graph.h"
#include "program.h"

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
    verbose(false), instrument(false), 
    warnings(false), program(NULL), graph(NULL),
    worker_threads(NULL), pending_count(0)
{
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
    if (!strcmp(argv[i],"-f"))
    {
      file_name = argv[++i];
      continue;
    }
    if (!strcmp(argv[i],"-i"))
    {
      instrument = true;
      continue;
    }
    if (!strcmp(argv[i],"-n"))
    {
      max_num_threads = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i],"-t"))
    {
      thread_pool_size = atoi(argv[++i]);
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
    fprintf(stdout,"  Max Number of Threads: %d\n", max_num_threads);
    fprintf(stdout,"  Thread Pool Size: %d\n", thread_pool_size);
    fprintf(stdout,"  Verbose: %s\n", (verbose ? "yes" : "no"));
    fprintf(stdout,"  Instrument: %s\n", (instrument ? "yes" : "no"));
    fprintf(stdout,"  Report Warnings: %s\n", (warnings ? "yes" : "no"));
  }
}

void Weft::report_usage(int error, const char *error_str)
{
  fprintf(stderr,"WEFT ERROR %d: %s\n! WEFT WILL NOW EXIT...\n", 
          error, error_str);
  fprintf(stderr,"Usage: Weft [args]\n");
  fprintf(stderr,"  -f: specify the input file\n");
  fprintf(stderr,"  -i: instrument execution\n");
  fprintf(stderr,"  -n: maximum number of threads per CTA\n");
  fprintf(stderr,"  -t: thread pool size\n");
  fprintf(stderr,"  -v: print verbose output\n");
  fprintf(stderr,"  -w: report emulation warnings\n");
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
  program->parse_ptx_file(file_name, max_num_threads);
  if (max_num_threads <= 0)
  {
    char buffer[1024];
    snprintf(buffer, 1023," Failed to find max number of threads "
                   "in file %s and the value was not set on the command "
                   "line using the '-n' flag", file_name);
    report_error(WEFT_ERROR_NO_THREAD_COUNT, buffer);
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
  assert(max_num_threads > 0);
  threads.resize(max_num_threads, NULL);
  initialize_count(max_num_threads);
  for (int i = 0; i < max_num_threads; i++)
  {
    threads[i] = new Thread(i, program); 
    EmulateTask *task = new EmulateTask(threads[i]);
    enqueue_task(task);
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

  // First initialization all the data structures
  initialize_count(threads.size());
  for (std::vector<Thread*>::const_iterator it = threads.begin();
        it != threads.end(); it++)
    enqueue_task(new InitializationTask(*it, threads.size(), max_num_barriers));
  wait_until_done();

  // Compute barrier reachability
  int total_barriers = graph->count_total_barriers();
  initialize_count(2*total_barriers);
  graph->enqueue_reachability_tasks();
  wait_until_done();

  // Compute latest/earliest happens-before/after tasks
  // There are twice as many of these as barriers
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

  if (instrument)
    stop_instrumentation(4/*stage*/);
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
  PTHREAD_SAFE_CALL( pthread_mutex_lock(&queue_lock) );
  if (queue.empty()) 
  {
    PTHREAD_SAFE_CALL( pthread_cond_wait(&queue_cond, &queue_lock) );
    // Check to see if the queue is still empty after waiting
    // If it is then we know we are done
    if (!queue.empty())
    {
      result = queue.front();
      queue.pop_front();
    }
  }
  else
  {
    result = queue.front();
    queue.pop_front();
  }
  PTHREAD_SAFE_CALL( pthread_mutex_unlock(&queue_lock) );
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
    fprintf(stdout,"  %50s: %10.3lf ms %12ld KB\n",
            stage_names[i], time, memory / 1024);
    total_time += timing[i];
    total_memory += memory;
  }
  fprintf(stdout,"  %50s: %10.3lf ms %12ld KB\n",
          "Total", double(total_time) * 1e-3, total_memory / 1024);
}

int main(int argc, char **argv)
{
  Weft weft(argc, argv);
  weft.verify();
  return 0;
}

