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

#ifndef __PROGRAM_H__
#define __PROGRAM_H__

#include <map>
#include <deque>
#include <vector>
#include <cassert>

class Weft;
class Thread;
class Happens;
class WeftAccess;
class SharedMemory;
class PTXInstruction;
class WeftInstruction;

class Program {
public:
  Program(Weft *weft);
  Program(const Program &rhs);
  ~Program(void);
public:
  Program& operator=(const Program &rhs);
public:
  void parse_ptx_file(const char *file_name, int &max_num_threads);
  void report_statistics(void);
  void report_statistics(const std::vector<Thread*> &threads);
  inline int count_instructions(void) const { return ptx_instructions.size(); }
public:
  int emulate(Thread *thread);
protected:
  void convert_to_instructions(int max_num_threads,
          const std::vector<std::pair<std::string,int> > &lines);
public:
  Weft *const weft;
protected:
  std::vector<PTXInstruction*> ptx_instructions;
};

class Thread {
public:
  Thread(unsigned thread_id, Program *p, SharedMemory *s);
  Thread(const Thread &rhs) : thread_id(0), 
    program(NULL), shared_memory(NULL) { assert(false); }
  ~Thread(void);
public:
  Thread& operator=(const Thread &rhs) { assert(false); return *this; }
public:
  void emulate(void);
public:
  void register_shared_location(const std::string &name, int64_t address);
  bool find_shared_location(const std::string &name, int64_t &addr);
public:
  void set_value(int64_t reg, int64_t value);
  bool get_value(int64_t reg, int64_t &value);
public:
  void set_pred(int64_t pred, bool value);
  bool get_pred(int64_t pred, bool &value);
public:
  void add_instruction(WeftInstruction *instruction);
  void update_max_barrier_name(int name);
  inline int get_max_barrier_name(void) const { return max_barrier_name; }
public:
  void profile_instruction(PTXInstruction *instruction);
  int accumulate_instruction_counts(std::vector<int> &total_counts);
public:
  void update_shared_memory(WeftAccess *access);
public:
  inline size_t get_program_size(void) const { return instructions.size(); }
  inline WeftInstruction* get_instruction(int idx)
    { return ((idx < instructions.size()) ? instructions[idx] : NULL); } 
  inline int count_dynamic_instructions(void) const 
    { return dynamic_instructions; }
public:
  void initialize_happens(int total_threads, int max_num_barriers);
  void update_happens_relationships(void);
protected:
  void initialize_happens_instances(int total_threads);
  void compute_barriers_before(int max_num_barriers);
  void compute_barriers_after(int max_num_barriers);
public:
  const unsigned thread_id;
  Program *const program;
  SharedMemory *const shared_memory;
protected:
  std::map<std::string,int64_t/*addr*/>           shared_locations;
  std::map<int64_t/*register*/,int64_t/*value*/>  register_store;
  std::map<int64_t/*predicate*/,bool/*value*/>    predicate_store;
protected:
  int max_barrier_name;
  int dynamic_instructions;
  std::vector<WeftInstruction*>                   instructions;
  std::vector<int>                                dynamic_counts;
protected:
  std::deque<Happens*>                            all_happens;
};

#endif //__PROGRAM_H__
