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
  : file_name(NULL), thread_pool_size(1), 
    verbose(false), detailed(false), instrument(false), 
    warnings(false), warp_synchronous(false), print_files(false),
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
  for (std::vector<Program*>::iterator it = programs.begin();
        it != programs.end(); it++)
  {
    delete (*it);
  }
  programs.clear();
}

void Weft::verify(void)
{
  Program::parse_ptx_file(file_name, this, programs);
  for (std::vector<Program*>::const_iterator it = programs.begin();
        it != programs.end(); it++)
  {
    Program *program = *it;
    program->verify(); 
  }
  if (instrument)
    report_instrumentation();
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
    if (!strcmp(argv[i],"-n"))
    {
      std::string threads(argv[++i]);
      parse_triple(threads, block_dim, "-n", "CTA size");
      continue;
    }
    if (!strcmp(argv[i],"-p"))
    {
      print_files = true;
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
  fprintf(stderr,"  -n: number of threads per CTA\n");
  fprintf(stderr,"      can be an integer or an x-separated tuple e.g. 64x2 or 32x8x1\n");
  fprintf(stderr,"  -p: print individual Weft thread files (one file per thread!)\n");
  fprintf(stderr,"  -s: assume warp-synchronous execution\n");
  fprintf(stderr,"  -t: thread pool size\n");
  fprintf(stderr,"  -v: print verbose output\n");
  fprintf(stderr,"  -w: report emulation warnings (this may generate considerable output)\n");
  exit(error);
}

bool Weft::initialize_program(Program *program) const
{
  program->set_block_dim(block_dim);
  program->add_block_id(block_id);
  program->set_grid_dim(grid_dim);
  return warp_synchronous;
}

void Weft::start_parsing_instrumentation(void)
{
  parsing_time = get_current_time_in_micros();
}

void Weft::stop_parsing_instrumentation(void)
{
  unsigned long long stop = get_current_time_in_micros();
  unsigned long long start = parsing_time;
  parsing_time = stop - start;
  parsing_memory = get_memory_usage();
}

void Weft::report_instrumentation(void)
{
  fprintf(stdout,"WEFT INSTRUMENTATION FOR PARSING FILE %s\n", file_name); 
#ifdef __MACH__
  fprintf(stdout,"  %50s: %10.3lf ms %12ld MB\n",
          "Parse PTX", double(parsing_time) * 1e-3, parsing_memory / (1024 * 1024));
#else
  fprintf(stdout,"  %50s: %10.3lf ms %12ld MB\n",
          "Parse PTX", double(parsing_time) * 1e-3, parsing_memory / 1024);
#endif
  size_t accumulated_memory = parsing_memory;
  for (std::vector<Program*>::const_iterator it = programs.begin();
        it != programs.end(); it++)
  {
    (*it)->report_instrumentation(accumulated_memory);
  }
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

int main(int argc, char **argv)
{
  Weft weft(argc, argv);
  weft.verify();
  fflush(stderr);
  fflush(stdout);
  return 0;
}

