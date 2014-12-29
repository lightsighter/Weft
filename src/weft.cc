
#include "weft.h"
#include "program.h"

#include <string>

#include <cstdio>
#include <cassert>
#include <cstring>
#include <cstdlib>

Weft::Weft(void)
  : file_name(NULL), max_num_threads(-1), 
    thread_pool_size(1), verbose(false)
{
}

void Weft::verify(int argc, char **argv)
{
  parse_inputs(argc, argv);  
  Program program;
  parse_ptx(program);
  assert(max_num_threads > 0); 
  std::vector<Thread> threads(max_num_threads);
  emulate_threads(program, threads);
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
  }
}

void Weft::report_usage(int error, const char *error_str)
{
  fprintf(stderr,"WEFT ERROR %d: %s! WEFT WILL NOW EXIT!\n", error, error_str);
  fprintf(stderr,"Usage: Weft [args]\n");
  fprintf(stderr,"  -f: specify the input file\n");
  fprintf(stderr,"  -n: maximum number of threads per CTA\n");
  fprintf(stderr,"  -t: thread pool size\n");
  fprintf(stderr,"  -v: print verbose output\n");
  exit(error);
}

void Weft::parse_ptx(Program &program)
{
  assert(file_name != NULL);
  if (verbose)
    fprintf(stdout,"WEFT INFO: Parsing file %s...\n", file_name);
  max_num_threads = program.parse_ptx_file(file_name, max_num_threads);
  if (max_num_threads <= 0)
  {
    fprintf(stderr,"WEFT ERROR %d: Failed to find max number of threads "
                   "in file %s and the value was not set on the command "
                   "line using the '-n' flag!\n",
                   WEFT_ERROR_NO_THREAD_COUNT, file_name);
    exit(WEFT_ERROR_NO_THREAD_COUNT);
  }
  if (verbose)
    program.report_statistics();
}

void Weft::emulate_threads(Program &program, std::vector<Thread> &threads)
{
  if (verbose)
    fprintf(stdout,"WEFT INFO: Emulating %d GPU threads "
                   "with %d CPU threads...\n",
                   max_num_threads, thread_pool_size);
}

int main(int argc, char **argv)
{
  Weft weft;
  weft.verify(argc, argv);
  return 0;
}

