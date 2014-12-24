
#ifndef __WEFT_H__
#define __WEFT_H__

#include <vector>

enum {
  WEFT_ERROR_NO_FILE_NAME,
  WEFT_ERROR_FILE_OPEN,
  WEFT_ERROR_THREAD_COUNT_MISMATCH,
  WEFT_ERROR_NO_THREAD_COUNT,
};

class Program;
class Thread;

class Weft {
public:
  Weft(void);
public:
  void verify(int argc, char **argv);
protected:
  void parse_inputs(int argc, char **argv);
  void report_usage(int error, const char *error_str);
  void parse_ptx(Program &p);
  void emulate_threads(Program &p, std::vector<Thread> &threads); 
protected:
  const char *file_name;
  int max_num_threads;
  int thread_pool_size;
  bool verbose;
};

#endif // __WEFT_H__
