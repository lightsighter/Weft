
#ifndef __PROGRAM_H__
#define __PROGRAM_H__

class Thread;
class PTXInstruction;

#include <map>
#include <vector>
#include <cassert>

class Program {
public:
  Program(void);
  Program(const Program &rhs);
  ~Program(void);
public:
  Program& operator=(const Program &rhs);
public:
  int parse_ptx_file(const char *file_name, int max_num_threads);
  void report_statistics(void);
public:
  void emulate(Thread *thread);
protected:
  void convert_to_instructions(int max_num_threads,
          const std::vector<std::pair<std::string,int> > &lines);
protected:
  std::vector<PTXInstruction*> ptx_instructions;
};

class Thread {
public:
  Thread(unsigned thread_id, Program *p);
  Thread(const Thread &rhs) : thread_id(0), program(NULL) { assert(false); }
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
  const unsigned thread_id;
  Program *const program;
protected:
  std::map<std::string,int64_t/*addr*/>           shared_locations;
  std::map<int64_t/*register*/,int64_t/*value*/>  register_store;
  std::map<int64_t/*predicate*/,bool/*value*/>    predicate_store;
};

#endif //__PROGRAM_H__
