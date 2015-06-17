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

#include <fstream>
#include <iostream>
#include <vector>

#include <cstdio>
#include <cstring>
#include <cassert>
#include <cstdlib>
#include <cxxabi.h> // Demangling

Program::Program(Weft *w, std::string &name)
  : weft(w), kernel_name(name), 
    max_num_threads(-1), max_num_barriers(1),
    current_cta(0)
{
  // Initialize values
  warp_synchronous = weft->initialize_program(this);
  max_num_threads = block_dim[0] * block_dim[1] * block_dim[2];
}

Program::Program(const Program &rhs)
  : weft(NULL)
{
  // should never be called
  assert(false);
}

Program::~Program(void)
{
  for (std::vector<PTXInstruction*>::iterator it = 
        ptx_instructions.begin(); it != ptx_instructions.end(); it++)
  {
    delete (*it);
  }
  ptx_instructions.clear();
  for (unsigned idx = 0; idx < cta_states.size(); idx++)
  {
    CTAState &state = cta_states[idx];
    if (state.shared_memory != NULL)
    {
      delete state.shared_memory;
      state.shared_memory = NULL;
    }
    if (state.graph != NULL)
    {
      delete state.graph;
      state.graph = NULL;
    }
    for (std::vector<Thread*>::iterator it = state.threads.begin();
          it != state.threads.end(); it++)
    {
      delete (*it);
    }
  }
  cta_states.clear();
}

Program& Program::operator=(const Program &rhs)
{
  // should never be called
  assert(false);
  return *this;
}

/*static*/
void Program::parse_ptx_file(const char *file_name, Weft *weft,
                             std::vector<Program*> &programs)
{
  assert(file_name != NULL);

  if (weft->print_verbose())
    fprintf(stdout,"WEFT INFO: Parsing file %s...\n", file_name);

  if (weft->perform_instrumentation())
    weft->start_parsing_instrumentation();

  Program *current = NULL;
  std::ifstream file(file_name);
  std::map<int,const char*> source_files;
  // First, let's get all the lines we care about
  if (!file.is_open())
  {
    char buffer[1024];
    snprintf(buffer, 1023, "Unable to open file %s", file_name);
    weft->report_error(WEFT_ERROR_FILE_OPEN, buffer);
  }
  bool found = false;
  std::string line, kernel_name;
  int line_num = 1;
  std::getline(file, line);
  int block_dim[3];
  while (!file.eof())
  {
    if (current != NULL)
      current->add_line(line, line_num);
    // Try parsing this as a file location
    parse_file_location(line, source_files);
    if (line.find(".entry") != std::string::npos)
    {
      // We should only have one entry kernel, we don't know
      // how to do this for more than one kernel at the moment
      if (current != NULL)
      {
        if (!found)
        {
          char buffer[1024];
          snprintf(buffer, 1023," Failed to find max number of threads for "
                         "kernel %s and the value was not set on the command "
                         "line using the '-n' flag", kernel_name.c_str());
          weft->report_error(WEFT_ERROR_NO_THREAD_COUNT, buffer);
        }
        else
        {
          current->set_block_dim(block_dim);
          found = false;
        }
        programs.push_back(current);
      }
      size_t start = line.find(".entry")+7;
      if (line.find("(") != std::string::npos)
      {
        size_t stop = line.find("(");
        int status;
        char *realname = 
          abi::__cxa_demangle(line.substr(start,stop-start).c_str(), 0, 0, &status);
        std::string full_name(realname);
        kernel_name = full_name.substr(0, full_name.find("("));
        current = new Program(weft, kernel_name);
        free(realname);
      }
      else
      {
        int status;
        char *realname = 
          abi::__cxa_demangle(line.substr(start).c_str(), 0, 0, &status);
        std::string full_name(realname);
        kernel_name = full_name.substr(0, full_name.find("("));
        current = new Program(weft, kernel_name);  
        free(realname);
      }
    }
    if (!found && (line.find(".maxntid") != std::string::npos))
    {
      found = true;
      std::string remaining = line.substr(line.find(".maxntid")+8);
      int str_index = 0;
      for (int i = 0; i < 3; i++)
      {
        bool has_dim = false;
        char buffer[8]; // No more than 1024 threads per CTA
        int buffer_index = 0; 
        while ((str_index < remaining.size()) &&
               ((remaining[str_index] < '0') || 
               (remaining[str_index] > '9')))
          str_index++; 
        while ((str_index < remaining.size()) &&
               (remaining[str_index] >= '0') &&
               (remaining[str_index] <= '9'))
        {
          buffer[buffer_index] = remaining[str_index];
          str_index++; buffer_index++;
          has_dim = true;
        }
        buffer[buffer_index] = '\0';
        if (has_dim)
          block_dim[i] = atoi(buffer);
        else
          block_dim[i] = 1; // Unspecified dimensions are set to one
      }
    }
    if (line.find(".version") != std::string::npos)
    {
      double version = atof(line.substr(line.find(" ")).c_str());
      if (version < 3.2)
      {
        char buffer[1024];
        snprintf(buffer,1023, "Weft requires PTX version 3.2 (CUDA 5.5) or later! "
                  "File %s contains PTX version %g", file_name, version);
        weft->report_error(WEFT_ERROR_INVALID_PTX_VERSION, buffer);
      }
    }
    line_num++;
    std::getline(file, line);
  }
  if (current == NULL)
  {
    char buffer[1024];
    snprintf(buffer,1023, "Weft found no entry point kernels "
                          "in PTX file %s\n", file_name);
    weft->report_error(WEFT_ERROR_NO_KERNELS, buffer);
  }
  else
    programs.push_back(current);
  if (!found)
  {
    char buffer[1024];
    snprintf(buffer, 1023," Failed to find max number of threads for "
                   "kernel %s and the value was not set on the command "
                   "line using the '-n' flag", kernel_name.c_str());
    weft->report_error(WEFT_ERROR_NO_THREAD_COUNT, buffer);  
  }
  else
    current->set_block_dim(block_dim);
  // If we didn't find a source file issue a warning
  if (source_files.empty())
    fprintf(stderr,"WEFT WARNING: No line information found! Line numbers from PTX "
       "will be used!\n\t\tTry re-running nvcc with the '-lineinfo' flag!\n");
  // Once we have the lines, then convert them into static PTX instructions
  for (std::vector<Program*>::const_iterator it = programs.begin();
        it != programs.end(); it++)
  {
    (*it)->convert_to_instructions(source_files);
  }

  if (weft->perform_instrumentation())
    weft->stop_parsing_instrumentation();

  if (weft->print_verbose())
  {
    for (std::vector<Program*>::const_iterator it = programs.begin();
          it != programs.end(); it++)
    {
      (*it)->report_statistics();
    }
  }
}

void Program::report_statistics(void)
{
  fprintf(stdout,"WEFT INFO: Program Statistics for Kernel %s\n", kernel_name.c_str());
  fprintf(stdout,"  Static Instructions: %ld\n", ptx_instructions.size());
  fprintf(stdout,"  Instruction Counts\n");
  unsigned counts[PTX_LAST];
  for (unsigned idx = 0; idx < PTX_LAST; idx++)
    counts[idx] = 0;
  for (std::vector<PTXInstruction*>::const_iterator it = 
        ptx_instructions.begin(); it != ptx_instructions.end(); it++)
    counts[(*it)->get_kind()]++;
  for (unsigned idx = 0; idx < PTX_LAST; idx++)
  {
    if (counts[idx] == 0)
      continue;
    fprintf(stdout,"    Instruction %s: %d\n", 
                   PTXInstruction::get_kind_name((PTXKind)idx), counts[idx]);
  }
  fprintf(stdout,"\n");
}

void Program::report_statistics(const std::vector<Thread*> &threads)
{
  int total_count = 0;
  std::vector<int> instruction_counts(PTX_LAST, 0);
  for (std::vector<Thread*>::const_iterator it = threads.begin();
        it != threads.end(); it++)
  {
    total_count += (*it)->accumulate_instruction_counts(instruction_counts);
  }
  fprintf(stdout,"WEFT INFO: Program Statistics for Kernel %s\n", kernel_name.c_str());
  fprintf(stdout,"  Dynamic Instructions: %d\n", total_count);
  fprintf(stdout,"  Instruction Counts\n");
  for (unsigned idx = 0; idx < PTX_LAST; idx++)
  {
    if (instruction_counts[idx] == 0)
      continue;
    fprintf(stdout,"    Instruction %s: %d\n", 
       PTXInstruction::get_kind_name((PTXKind)idx), instruction_counts[idx]);
  }
  fprintf(stdout,"\n");
}

bool Program::has_shuffles(void) const
{
  for (std::vector<PTXInstruction*>::const_iterator it = 
        ptx_instructions.begin(); it != ptx_instructions.end(); it++)
  {
    if ((*it)->is_shuffle())
      return true;
  }
  return false;
}

void Program::emulate_threads(void)
{
  if (weft->print_verbose())
    fprintf(stdout,"WEFT INFO: Emulating %d GPU threads "
                   "for kernel %s...\n",
                   max_num_threads, kernel_name.c_str());
  
  if (weft->perform_instrumentation())
    start_instrumentation(EMULATE_THREADS_STAGE);

  SharedMemory *&shared_memory = cta_states[current_cta].shared_memory;
  assert(shared_memory == NULL);
  shared_memory = new SharedMemory(weft, this);
  assert(max_num_threads > 0);
  assert(max_num_threads == (block_dim[0]*block_dim[1]*block_dim[2]));
  std::vector<Thread*> &threads = cta_states[current_cta].threads;
  threads.resize(max_num_threads, NULL);
  // If we are doing warp synchronous execution we 
  // execute all the threads in a warp together
  if (warp_synchronous) 
  {
    assert((max_num_threads % WARP_SIZE) == 0);
    weft->initialize_count(max_num_threads/WARP_SIZE);
    int tid = 0;
    for (int z = 0; z < block_dim[2]; z++)
    {
      for (int y = 0; y < block_dim[1]; y++)
      {
        for (int x = 0; x < block_dim[0]; x++)
        {
          threads[tid] = new Thread(tid, x, y, z, this, shared_memory);
          // Increment first
          tid++;
          // Only kick off a warp once we've generated all the threads
          if ((tid % WARP_SIZE) == 0)
          {
            assert((tid-WARP_SIZE) >= 0);
            EmulateWarp *task = 
              new EmulateWarp(this, &(threads[tid-WARP_SIZE]));
            weft->enqueue_task(task);
          }
        }
      }
    }
  }
  else
  {
    weft->initialize_count(max_num_threads);
    int tid = 0;
    for (int z = 0; z < block_dim[2]; z++)
    {
      for (int y = 0; y < block_dim[1]; y++)
      {
        for (int x = 0; x < block_dim[0]; x++)
        {
          threads[tid] = new Thread(tid, x, y, z, this, shared_memory); 
          EmulateThread *task = new EmulateThread(threads[tid]);
          weft->enqueue_task(task);
          tid++;
        }
      }
    }
  }
  weft->wait_until_done();
  // Get the maximum barrier ID from all threads
  for (int i = 0; i < max_num_threads; i++)
  {
    int local_max = threads[i]->get_max_barrier_name();
    if ((local_max+1) > max_num_barriers)
      max_num_barriers = (local_max+1);
  }
  if (weft->print_verbose())
  {
    fprintf(stdout,"WEFT INFO: Emulation found %d named barriers for kernel %s.\n",
                    barrier_upper_bound(), kernel_name.c_str());
    report_statistics();
  }

  if (weft->perform_instrumentation())
    stop_instrumentation(EMULATE_THREADS_STAGE);

  // If we want to dump thread-specific files, do that now
  // Note that we don't include this in the timing
  if (weft->emit_program_files())
    print_files();
}

void Program::construct_dependence_graph(void)
{
  if (weft->print_verbose())
    fprintf(stdout,"WEFT INFO: Constructing barrier dependence graph "
                   "for kernel %s...\n", kernel_name.c_str());

  if (weft->perform_instrumentation())
    start_instrumentation(CONSTRUCT_BARRIER_GRAPH_STAGE);

  BarrierDependenceGraph *&graph = cta_states[current_cta].graph;
  assert(graph == NULL);
  graph = new BarrierDependenceGraph(weft, this);
  std::vector<Thread*> &threads = cta_states[current_cta].threads;
  graph->construct_graph(threads);

  // Validate the graph 
  int total_validation_tasks = graph->count_validation_tasks();
  if (weft->print_verbose())
    fprintf(stdout,"WEFT INFO: Performing %d graph validation checks...\n",
                              total_validation_tasks);
  if (total_validation_tasks > 0)
  {
    weft->initialize_count(total_validation_tasks);
    graph->enqueue_validation_tasks();
    weft->wait_until_done();
    graph->check_for_validation_errors();
  }

  if (weft->perform_instrumentation())
    stop_instrumentation(CONSTRUCT_BARRIER_GRAPH_STAGE);
}

void Program::compute_happens_relationships(void)
{
  if (weft->print_verbose())
    fprintf(stdout,"WEFT INFO: Computing happens-before/after "
                   "relationships for kernel %s...\n", kernel_name.c_str());

  if (weft->perform_instrumentation())
    start_instrumentation(COMPUTE_HAPPENS_RELATIONSHIP_STAGE);

  // First initialize all the data structures
  std::vector<Thread*> &threads = cta_states[current_cta].threads;
  weft->initialize_count(threads.size());
  for (std::vector<Thread*>::const_iterator it = threads.begin();
        it != threads.end(); it++)
    weft->enqueue_task(
        new InitializationTask(*it, threads.size(), max_num_barriers));
  weft->wait_until_done();

  // Compute barrier reachability
  // There are twice as many tasks as barriers
  BarrierDependenceGraph *&graph = cta_states[current_cta].graph;
  int total_barriers = graph->count_total_barriers();
  weft->initialize_count(2*total_barriers);
  graph->enqueue_reachability_tasks();
  weft->wait_until_done();

  // Compute latest/earliest happens-before/after tasks
  // There are twice as many tasks as barriers
  weft->initialize_count(2*total_barriers);
  graph->enqueue_transitive_happens_tasks();
  weft->wait_until_done();

  // Finally update all the happens relationships
  weft->initialize_count(threads.size());
  for (std::vector<Thread*>::const_iterator it = threads.begin();
        it != threads.end(); it++)
    weft->enqueue_task(new UpdateThreadTask(*it));
  weft->wait_until_done();

  if (weft->perform_instrumentation())
    stop_instrumentation(COMPUTE_HAPPENS_RELATIONSHIP_STAGE);
}

void Program::check_for_race_conditions(void)
{
  if (weft->print_verbose())
    fprintf(stdout,"WEFT INFO: Checking for race conditions in "
                   "kernel %s...\n", kernel_name.c_str());

  if (weft->perform_instrumentation())
    start_instrumentation(CHECK_FOR_RACES_STAGE);

  SharedMemory *&shared_memory = cta_states[current_cta].shared_memory;
  weft->initialize_count(shared_memory->count_addresses());
  shared_memory->enqueue_race_checks();
  weft->wait_until_done();
  shared_memory->check_for_races();

  if (weft->perform_instrumentation())
    stop_instrumentation(CHECK_FOR_RACES_STAGE);
}

void Program::print_statistics(void)
{
  fprintf(stdout,"WEFT STATISTICS for Kernel %s\n", kernel_name.c_str());
  fprintf(stdout,"  CTA Thread Count:          %15d\n", max_num_threads);
  fprintf(stdout,"  Shared Memory Locations:   %15d\n", count_addresses());
  fprintf(stdout,"  Physical Named Barriers;   %15d\n", max_num_barriers);
  fprintf(stdout,"  Dynamic Barrier Instances: %15d\n", count_total_barriers());
  fprintf(stdout,"  Static Instructions:       %15d\n", count_instructions());
  fprintf(stdout,"  Dynamic Instructions:      %15d\n", count_dynamic_instructions());
  fprintf(stdout,"  Weft Statements:           %15d\n", count_weft_statements());   
  fprintf(stdout,"  Total Race Tests:          %15ld\n",count_race_tests());
}

void Program::print_files(void)
{
  // We'll only dump the first CTA worth of threads for now
  assert(!cta_states.empty());
  std::vector<Thread*> &threads = cta_states[0].threads;
  weft->initialize_count(max_num_threads);
  for (std::vector<Thread*>::const_iterator it = threads.begin();
        it != threads.end(); it++)
  {
    DumpThreadTask *dump_task = new DumpThreadTask(*it); 
    weft->enqueue_task(dump_task);
  }
  weft->wait_until_done();
}

int Program::count_dynamic_instructions(void)
{
  int result = 0;
  for (unsigned idx = 0; idx < cta_states.size(); idx++)
  {
    std::vector<Thread*> &threads = cta_states[idx].threads;
    for (std::vector<Thread*>::const_iterator it = threads.begin();
          it != threads.end(); it++)
    {
      result += (*it)->count_dynamic_instructions();
    }
  }
  return result;
}

int Program::count_weft_statements(void)
{
  int result = 0;
  for (unsigned idx = 0; idx < cta_states.size(); idx++)
  {
    std::vector<Thread*> &threads = cta_states[idx].threads;
    for (std::vector<Thread*>::const_iterator it = threads.begin();
          it != threads.end(); it++)
    {
      result += (*it)->count_weft_statements();
    }
  }
  return result;
}

int Program::count_total_barriers(void)
{
  int result = 0;
  for (unsigned idx = 0; idx < cta_states.size(); idx++)
  {
    result += cta_states[idx].graph->count_total_barriers();
  }
  return result;
}

int Program::count_addresses(void)
{
  int result = 0;
  for (unsigned idx = 0; idx < cta_states.size(); idx++)
  {
    result += cta_states[idx].shared_memory->count_addresses();
  }
  return result;
}

size_t Program::count_race_tests(void)
{
  size_t result = 0;
  for (unsigned idx = 0; idx < cta_states.size(); idx++)
  {
    result += cta_states[idx].shared_memory->count_race_tests();
  }
  return result;
}

int Program::emulate(Thread *thread)
{
  int dynamic_instructions = 0;
  PTXInstruction *pc = ptx_instructions.front();
  bool profile = weft->print_verbose();
  if (profile)
  {
    while (pc != NULL)
    {
      thread->profile_instruction(pc);
      pc = pc->emulate(thread);
      dynamic_instructions++;
    }
  }
  else
  {
    while (pc != NULL)
    {
      pc = pc->emulate(thread);
      dynamic_instructions++;
    }
  }
  return dynamic_instructions;
}

void Program::emulate_warp(Thread **threads)
{
  // Execute all the threads in lock-step
  PTXInstruction *pc = ptx_instructions.front();  
  ThreadState thread_state[WARP_SIZE];
  for (int i = 0; i < WARP_SIZE; i++)
    thread_state[i] = ThreadState();
  int dynamic_instructions[WARP_SIZE];
  for (int i = 0; i < WARP_SIZE; i++)
    dynamic_instructions[i] = 0;
  int shared_access_id = 0;
  SharedStore store;
  bool profile = weft->print_verbose();
  if (profile)
  {
    while (pc != NULL)
    {
      for (int i = 0; i < WARP_SIZE; i++)
      {
        if (thread_state[i].status == THREAD_ENABLED)
        {
          threads[i]->profile_instruction(pc);
          dynamic_instructions[i]++;
        }
      }
      pc = pc->emulate_warp(threads, thread_state, 
                            shared_access_id, store);
    }
  }
  else
  {
    while (pc != NULL)
    {
      for (int i = 0; i < WARP_SIZE; i++)
      {
        if (thread_state[i].status == THREAD_ENABLED)
          dynamic_instructions[i]++;
      }
      pc = pc->emulate_warp(threads, thread_state, 
                            shared_access_id, store);
    }
  }
  for (int i = 0; i < WARP_SIZE; i++)
    threads[i]->set_dynamic_instructions(dynamic_instructions[i]);
}

void Program::get_kernel_prefix(char *buffer, size_t count)
{
  strncpy(buffer, kernel_name.c_str(), count);
}

void Program::add_line(const std::string &line, int line_num)
{
  lines.push_back(std::pair<std::string,int>(line, line_num));
}

void Program::set_block_dim(const int *array)
{
  for (int i = 0; i < 3; i++)
    block_dim[i] = array[i];
  max_num_threads = block_dim[0] * block_dim[1] * block_dim[2];
}

void Program::add_block_id(const int *array)
{
  cta_states.push_back(CTAState()); 
  CTAState &last = cta_states.back();
  for (int i = 0; i < 3; i++)
    last.block_id[i] = array[i];
}

void Program::set_grid_dim(const int *array)
{
  for (int i = 0; i < 3; i++)
    grid_dim[i] = array[i];
}

void Program::fill_block_dim(int *array) const
{
  for (int i = 0; i < 3; i++)
    array[i] = block_dim[i];
}

void Program::fill_block_id(int *array) const
{
  for (int i = 0; i < 3; i++)
    array[i] = block_id[i];
}

void Program::fill_grid_dim(int *array) const
{
  for (int i = 0; i < 3; i++)
    array[i] = grid_dim[i];
}

void Program::verify(void)
{
  emulate_threads();
  construct_dependence_graph();
  compute_happens_relationships();
  check_for_race_conditions();
  print_statistics();
}

void Program::convert_to_instructions(
                const std::map<int,const char*> &source_files)
{
  // Make a first pass and create all the instructions
  // Track all the basic block program counters
  std::map<std::string,PTXLabel*> labels;
  PTXInstruction *previous = NULL;
  int current_source_file = -1;
  int current_source_line = -1;
  for (std::vector<std::pair<std::string,int> >::const_iterator it = 
        lines.begin(); it != lines.end(); it++)
  {
    if (parse_source_location(it->first, current_source_file, current_source_line))
      continue;
    PTXInstruction *next = PTXInstruction::interpret(it->first, it->second);
    // Skip any empty lines
    if (next == NULL)
      continue;
    if (current_source_file >= 0)
    {
      std::map<int,const char*>::const_iterator finder = 
        source_files.find(current_source_file);
      assert(finder != source_files.end());
      next->set_source_location(finder->second, current_source_line);
    }
    ptx_instructions.push_back(next);
    if (next->is_label())
    {
      PTXLabel *label = next->as_label();
      label->update_labels(labels);
    }
    if (previous != NULL)
      previous->set_next(next);
    previous = next;
  } 
  // Then make a second pass to fill in the pointers
  for (std::vector<PTXInstruction*>::const_iterator it = 
        ptx_instructions.begin(); it != ptx_instructions.end(); it++)
  {
    if ((*it)->is_branch())
    {
      PTXBranch *branch = (*it)->as_branch();
      branch->set_targets(labels);
    }
    if ((*it)->is_barrier())
    {
      PTXBarrier *barrier = (*it)->as_barrier();
      barrier->update_count(max_num_threads);
    }
  }
  // Check for shuffles, if we have shuffles then make sure
  // that we have enabled warp-synchronous execution
  if (!warp_synchronous && has_shuffles())
  {
    fprintf(stdout,"WEFT WARNING: Program %s has shuffle instructions "
                   "but warp-synchronous execution was not assumed!\n"
                   "Enabling warp-synchronous assumption...\n",
                   kernel_name.c_str());
    warp_synchronous = true;
  }
  lines.clear();
}

/*static*/
bool Program::parse_file_location(const std::string &line,
                                  std::map<int,const char*> &source_files)
{
  if (line.find(".file") != std::string::npos)
  {
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    assert(tokens.size() == 5);
    int file_id = atoi(tokens[1].c_str());
    int start = tokens[2].find_last_of("/");
    int end = tokens[2].find("\"",1); 
    assert(source_files.find(file_id) == source_files.end());
    source_files[file_id] = strdup(tokens[2].substr(start+1,end-start-1).c_str());
    return true;
  }
  return false;
}

/*static*/
bool Program::parse_source_location(const std::string &line,
                                    int &source_file, int &source_line)
{
  if ((line.find(".loc") != std::string::npos) &&
      (line.find(".local") == std::string::npos))
  {
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    assert(tokens.size() == 4);
    source_file = atoi(tokens[1].c_str());
    source_line = atoi(tokens[2].c_str());
    return true;
  }
  return false;
}

void Program::start_instrumentation(ProgramStage stage)
{
  timing[stage] = weft->get_current_time_in_micros();
}

void Program::stop_instrumentation(ProgramStage stage)
{
  unsigned long long stop = weft->get_current_time_in_micros();
  unsigned long long start = timing[stage];
  timing[stage] = stop - start;
  memory_usage[stage] = weft->get_memory_usage();
}

void Program::report_instrumentation(size_t &accumulated_memory)
{
  const char *stage_names[TOTAL_STAGES] = { "Emulate Threads",
                         "Construct Barrier Dependence Graph",
                         "Compute Happens-Before/After Relationships",
                         "Check for Race Conditions" };
  fprintf(stdout,"WEFT INSTRUMENTATION FOR KERNEL %s\n", kernel_name.c_str());
  unsigned long long total_time = 0;
  size_t total_memory = 0;
  for (int i = 0; i < TOTAL_STAGES; i++)
  {
    double time = double(timing[i]) * 1e-3;
    size_t memory = memory_usage[i] - accumulated_memory;
#ifdef __MACH__
    fprintf(stdout,"  %50s: %10.3lf ms %12ld MB\n",
            stage_names[i], time, memory / (1024 * 1024));
#else
    fprintf(stdout,"  %50s: %10.3lf ms %12ld MB\n",
            stage_names[i], time, memory / 1024);
#endif
    total_time += timing[i];
    total_memory += memory;
    accumulated_memory += memory;
  }
#ifdef __MACH__
  fprintf(stdout,"  %50s: %10.3lf ms %12ld MB\n",
          "Total", double(total_time) * 1e-3, total_memory / (1024*1024));
#else
  fprintf(stdout,"  %50s: %10.3lf ms %12ld MB\n",
          "Total", double(total_time) * 1e-3, total_memory / 1024);
#endif
}

Thread::Thread(unsigned tid, int tidx, int tidy, int tidz,
               Program *p, SharedMemory *m)
  : thread_id(tid), tid_x(tidx), tid_y(tidy), tid_z(tidz),
    program(p), shared_memory(m), 
    max_barrier_name(-1), dynamic_instructions(0)
{
  dynamic_counts.resize(PTX_LAST, 0);
}

Thread::~Thread(void)
{
  // Clean up our instructions
  for (std::vector<WeftInstruction*>::iterator it = 
        instructions.begin(); it != instructions.end(); it++)
  {
    delete (*it);
  }
  instructions.clear();
  for (std::deque<Happens*>::iterator it = 
        all_happens.begin(); it != all_happens.end(); it++)
  {
    delete (*it);
  }
  all_happens.clear();
}

void Thread::initialize(void)
{
  int block_dim[3];
  int block_id[3];
  int grid_dim[3];
  program->fill_block_dim(block_dim);
  program->fill_block_id(block_id);
  program->fill_grid_dim(grid_dim);
  // Before starting emulation fill in the special
  // values for particular registers
  register_store[WEFT_TID_X_REG] = tid_x;
  register_store[WEFT_TID_Y_REG] = tid_y;
  register_store[WEFT_TID_Z_REG] = tid_z;
  register_store[WEFT_NTID_X_REG] = block_dim[0];
  register_store[WEFT_NTID_Y_REG] = block_dim[1];
  register_store[WEFT_NTID_Z_REG] = block_dim[2];
  register_store[WEFT_LANE_REG] = (thread_id % WARP_SIZE);
  register_store[WEFT_WARP_REG] = (thread_id / WARP_SIZE);
  register_store[WEFT_NWARP_REG] = 
    (block_dim[0] * block_dim[1] * block_dim[2] + (WARP_SIZE-1)) / WARP_SIZE;
  register_store[WEFT_CTA_X_REG] = block_id[0];
  register_store[WEFT_CTA_Y_REG] = block_id[1];
  register_store[WEFT_CTA_Z_REG] = block_id[2];
  register_store[WEFT_NCTA_X_REG] = grid_dim[0];
  register_store[WEFT_NCTA_Y_REG] = grid_dim[1];
  register_store[WEFT_NCTA_Z_REG] = grid_dim[2];
}

void Thread::emulate(void)
{
  dynamic_instructions = program->emulate(this);
}

void Thread::cleanup(void)
{
  // Once we are done we can clean up all our data structures
  shared_locations.clear();
  register_store.clear();
  predicate_store.clear();
  globals.clear();
}

void Thread::register_shared_location(const std::string &name, int64_t addr)
{
  assert(shared_locations.find(name) == shared_locations.end());
  shared_locations[name] = addr;
}

bool Thread::find_shared_location(const std::string &name, int64_t &addr)
{
  std::map<std::string,int64_t>::const_iterator finder = 
    shared_locations.find(name);
  if (finder == shared_locations.end()) {
    if (program->weft->report_warnings())
    {
      fprintf(stderr,"WEFT WARNING: Unable to find shared "
                     "memory location %s\n", name.c_str());
    }
    return false;
  }
  addr = finder->second;
  return true;
}

void Thread::register_global_location(const char *name, const int *data, size_t size)
{
  GlobalDataInfo info;
  info.name = name;
  info.data = data;
  info.size = size;
  globals.push_back(info);
}

bool Thread::get_global_location(const char *name, int64_t &value)
{
  for (unsigned idx = 0; idx < globals.size(); idx++)
  {
    // See if the names match
    if (strcmp(name, globals[idx].name) == 0)
    {
      value = idx * SDDRINC;
      return true;
    }
  }
  return false;
}

bool Thread::get_global_value(int64_t addr, int64_t &value)
{
  int index = addr / SDDRINC;
  if ((index >= 0) && (index < int(globals.size())))
  {
    size_t offset = addr - (index * SDDRINC);
    assert(offset < globals[index].size);
    value = globals[index].data[offset];
    return true;
  }
  return false;
}

void Thread::set_value(int64_t reg, int64_t value)
{
  register_store[reg] = value;
}

bool Thread::get_value(int64_t reg, int64_t &value)
{
  std::map<int64_t,int64_t>::const_iterator finder = 
    register_store.find(reg);
  if (finder == register_store.end()) {
    if (program->weft->report_warnings())
    {
      char buffer[11];
      PTXInstruction::decompress_identifier(reg, buffer, 11);
      fprintf(stderr,"WEFT WARNING: Unable to find register %s\n", buffer);
    }
    return false;
  }
  value = finder->second;
  return true;
}

void Thread::set_pred(int64_t pred, bool value)
{
  predicate_store[pred] = value;
}

bool Thread::get_pred(int64_t pred, bool &value)
{
  std::map<int64_t,bool>::const_iterator finder = 
    predicate_store.find(pred);
  if (finder == predicate_store.end()) {
    if (program->weft->report_warnings())
    {
      char buffer[11];
      PTXInstruction::decompress_identifier(pred, buffer, 11);
      fprintf(stderr,"WEFT WARNING: Unable to find predicate %s\n", buffer);
    }
    return false;
  }
  value = finder->second;
  return true;
}

void Thread::add_instruction(WeftInstruction *instruction)
{
  instructions.push_back(instruction);
}

void Thread::update_max_barrier_name(int name)
{
  if (name > max_barrier_name)
    max_barrier_name = name;
}

void Thread::profile_instruction(PTXInstruction *instruction)
{
  int kind = instruction->get_kind();  
  dynamic_counts[kind]++;
}

int Thread::accumulate_instruction_counts(std::vector<int> &total_counts)
{
  int total = 0; 
  assert(total_counts.size() == dynamic_counts.size());
  for (unsigned idx = 0; idx < total_counts.size(); idx++)
  {
    total_counts[idx] += dynamic_counts[idx];
    total += dynamic_counts[idx];
  }
  return total;
}

void Thread::dump_weft_thread(void)
{
  // Open up a file for this thread and then 
  // print out all of our weft instructions
  char file_name[1024];
  program->get_kernel_prefix(file_name, 1024-32);
  char buffer[32];
  snprintf(buffer, 31, "_%d_%d_%d.weft", tid_x, tid_y, tid_z);
  strncat(file_name, buffer, 31);
  FILE *weft_file = fopen(file_name, "w");
  if (weft_file == NULL)
  {
    fprintf(stderr, "WEFT WARNING: Failed to open file %s\n", file_name);
    return ;
  }
  for (std::vector<WeftInstruction*>::const_iterator it = 
        instructions.begin(); it != instructions.end(); it++)
  {
    (*it)->print_instruction(weft_file);
  }
  assert(fclose(weft_file) == 0);
}

void Thread::update_shared_memory(WeftAccess *access)
{
  shared_memory->update_accesses(access);
}

void Thread::initialize_happens(int total_threads,
                                int max_num_barriers)
{
  initialize_happens_instances(total_threads); 
  compute_barriers_before(max_num_barriers);
  compute_barriers_after(max_num_barriers);
}

void Thread::update_happens_relationships(void)
{
  for (std::deque<Happens*>::const_iterator it = 
        all_happens.begin(); it != all_happens.end(); it++)
  {
    (*it)->update_happens_relationships();
  }
}

void Thread::initialize_happens_instances(int total_threads)
{
  // First create
  Happens *next = NULL;  
  for (std::vector<WeftInstruction*>::const_iterator it = 
        instructions.begin(); it != instructions.end(); it++)
  {
    // Don't make happens for barriers
    if ((*it)->is_barrier())
    {
      next = NULL;
      continue;
    }
    if (next == NULL)
    {
      next = new Happens(total_threads);
      all_happens.push_back(next);
    }
    (*it)->initialize_happens(next);
  }
}

void Thread::compute_barriers_before(int max_num_barriers)
{
  std::vector<WeftBarrier*> before_barriers(max_num_barriers, NULL);
  bool has_update = false;
  for (std::vector<WeftInstruction*>::const_iterator it = 
        instructions.begin(); it != instructions.end(); it++)
  {
    // We only count syncs in the set of barriers before
    // because they are the only instructions which can 
    // establish a happens-before relationship. On the
    // contrary, arrives can always establish a happens-after.
    if ((*it)->is_sync())
    {
      WeftBarrier *bar = (*it)->as_barrier(); 
      assert(bar->name < max_num_barriers);
      before_barriers[bar->name] = bar;
      has_update = true;
    }
    else if ((*it)->is_arrive())
      has_update = true; // set to true to update next happens
    else if (has_update)
    {
      Happens *happens = (*it)->get_happens();
      assert(happens != NULL);
      happens->update_barriers_before(before_barriers);
      has_update = false;
    }
  }
}

void Thread::compute_barriers_after(int max_num_barriers)
{
  std::vector<WeftBarrier*> after_barriers(max_num_barriers, NULL);
  bool has_update = false;
  for (std::vector<WeftInstruction*>::reverse_iterator it = 
        instructions.rbegin(); it != instructions.rend(); it++)
  {
    if ((*it)->is_barrier())
    {
      WeftBarrier *bar = (*it)->as_barrier();
      assert(bar->name < max_num_barriers);
      after_barriers[bar->name] = bar;
      has_update = true;
    }
    else if (has_update)
    {
      Happens *happens = (*it)->get_happens();
      assert(happens != NULL);
      happens->update_barriers_after(after_barriers);
      has_update = false;
    }
  }
}

void SharedStore::write(int64_t addr, int64_t value)
{
  store[addr] = value;
}

bool SharedStore::read(int64_t addr, int64_t &value)
{
  std::map<int64_t,int64_t>::const_iterator finder = store.find(addr);
  if (finder == store.end())
    return false;
  value = finder->second;
  return true;
}

EmulateThread::EmulateThread(Thread *t)
  : WeftTask(), thread(t)
{
}

void EmulateThread::execute(void)
{
  thread->initialize();
  thread->emulate();
  thread->cleanup();
}

EmulateWarp::EmulateWarp(Program *p, Thread **start)
  : WeftTask(), program(p), threads(start)
{
}

void EmulateWarp::execute(void)
{
  // Initialize all the threads
  for (int i = 0; i < WARP_SIZE; i++)
    threads[i]->initialize();

  // Have the program simulate all the threads together
  program->emulate_warp(threads);

  // Cleanup all the threads
  for (int i = 0; i < WARP_SIZE; i++)
    threads[i]->cleanup();
}

InitializationTask::InitializationTask(Thread *t, int total, int max_barriers)
  : WeftTask(), thread(t), total_threads(total), max_num_barriers(max_barriers)
{
}

void InitializationTask::execute(void)
{
  thread->initialize_happens(total_threads, max_num_barriers);
}

UpdateThreadTask::UpdateThreadTask(Thread *t)
  : WeftTask(), thread(t)
{
}

void UpdateThreadTask::execute(void)
{
  thread->update_happens_relationships();
}

DumpThreadTask::DumpThreadTask(Thread *t)
  : WeftTask(), thread(t)
{
}

void DumpThreadTask::execute(void)
{
  thread->dump_weft_thread();
}

