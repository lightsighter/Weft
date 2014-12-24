
#ifndef __PROGRAM_H__
#define __PROGRAM_H__

class PTXInstruction;

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
protected:
  void convert_to_instructions(
          const std::vector<std::pair<std::string,int> > &lines);
protected:
  std::vector<PTXInstruction*> ptx_instructions;
};

class Thread {
public:
  Thread(void);
};

#endif //__PROGRAM_H__
