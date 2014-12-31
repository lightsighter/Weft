
#include "weft.h"
#include "program.h"
#include "instruction.h"

#include <fstream>
#include <iostream>
#include <vector>

#include <cstdio>
#include <cstring>
#include <cassert>
#include <cstdlib>

Program::Program(void)
{
}

Program::Program(const Program &rhs)
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
}

Program& Program::operator=(const Program &rhs)
{
  // should never be called
  assert(false);
  return *this;
}

int Program::parse_ptx_file(const char *file_name, int max_num_threads)
{
  std::ifstream file(file_name);
  std::vector<std::pair<std::string,int> > lines;
  // First, let's get all the lines we care about
  if (file.is_open())
  {
    bool start_recording = false;
    bool found = false;
    std::string line;
    int line_num = 0;
    while (std::getline(file, line))
    {
      if (start_recording)
        lines.push_back(std::pair<std::string,int>(line,line_num));
      if (line.find(".entry") != std::string::npos)
        start_recording = true;
      if (!found && (line.find(".maxntid") != std::string::npos))
      {
        int temp = atoi(line.substr(line.find(" "),line.find(",")).c_str());
        if (max_num_threads != -1)
        {
          if (temp != max_num_threads)
          {
            fprintf(stderr,"WEFT ERROR %d: Found max thread count %d "
                           "which does not agree with specified count "
                           "of %d!\n", WEFT_ERROR_THREAD_COUNT_MISMATCH,
                           temp, max_num_threads);
            exit(WEFT_ERROR_THREAD_COUNT_MISMATCH);
          }
        }
        else
          max_num_threads = temp;
        found = true;
      }
      line_num++;
    }
  }
  else
  {
    fprintf(stderr,"WEFT ERROR %d: unable to open file %s!\n",
            WEFT_ERROR_FILE_OPEN, file_name);
    exit(WEFT_ERROR_FILE_OPEN);
  }
  // Once we have the lines, then convert them into static PTX instructions
  convert_to_instructions(max_num_threads, lines);

  return max_num_threads;
}

void Program::report_statistics(void)
{
  fprintf(stdout,"WEFT INFO: Program Statistics\n");
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

void Program::emulate(Thread *thread)
{
  PTXInstruction *pc = ptx_instructions.front();
  while (pc != NULL)
    pc = pc->emulate(thread);
}

void Program::convert_to_instructions(int max_num_threads,
                const std::vector<std::pair<std::string,int> > &lines)
{
  // Make a first pass and create all the instructions
  // Track all the basic block program counters
  std::map<std::string,PTXInstruction*> labels;
  PTXInstruction *previous = NULL;
  for (std::vector<std::pair<std::string,int> >::const_iterator it = 
        lines.begin(); it != lines.end(); it++)
  {
    PTXInstruction *next = PTXInstruction::interpret(it->first, it->second);
    // Skip any empty lines
    if (next == NULL)
      continue;
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
}

Thread::Thread(unsigned tid, Program *p)
  : thread_id(tid), program(p)
{
}

Thread::~Thread(void)
{
}

void Thread::emulate(void)
{
  // Before starting emulation fill in the special
  // values for particular registers
  register_store[WEFT_TID_REG] = thread_id;
  // Use 0 as the default CTA ID
  register_store[WEFT_CTA_REG] = 0; 
  program->emulate(this);
  // Once we are done we can clean up all our data structures
  shared_locations.clear();
  register_store.clear();
  predicate_store.clear();
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
    fprintf(stderr,"WEFT WARNING: Unable to find shared "
                   "memory location %s\n", name.c_str());
    return false;
  }
  addr = finder->second;
  return true;
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
#if 0
    char buffer[11];
    PTXInstruction::decompress_identifier(reg, buffer, 11);
    fprintf(stderr,"WEFT WARNING: Unable to find register %s\n", buffer);
#endif
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
#if 0
    char buffer[11];
    PTXInstruction::decompress_identifier(pred, buffer, 11);
    fprintf(stderr,"WEFT WARNING: Unable to find predicate %s\n", buffer);
#endif
    return false;
  }
  value = finder->second;
  return true;
}

