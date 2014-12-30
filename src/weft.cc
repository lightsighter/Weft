
#include "weft.h"
#include "program.h"

#include <string>

#include <cstdio>
#include <cassert>
#include <cstring>
#include <cstdlib>

Weft::Weft(int argc, char **argv)
  : file_name(NULL), max_num_threads(-1), 
    thread_pool_size(1), verbose(false),
    program(NULL), worker_threads(NULL), 
    pending_count(0)
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
  for (std::vector<Thread*>::iterator it = threads.begin();
        it != threads.end(); it++)
  {
    delete (*it);
  }
  threads.clear();
}

void Weft::verify(void)
{
  parse_ptx();
  emulate_threads();
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

void Weft::parse_ptx(void)
{
  assert(file_name != NULL);
  if (verbose)
    fprintf(stdout,"WEFT INFO: Parsing file %s...\n", file_name);
  assert(program == NULL);
  program = new Program();
  max_num_threads = program->parse_ptx_file(file_name, max_num_threads);
  if (max_num_threads <= 0)
  {
    fprintf(stderr,"WEFT ERROR %d: Failed to find max number of threads "
                   "in file %s and the value was not set on the command "
                   "line using the '-n' flag!\n",
                   WEFT_ERROR_NO_THREAD_COUNT, file_name);
    exit(WEFT_ERROR_NO_THREAD_COUNT);
  }
  if (verbose)
    program->report_statistics();
}

void Weft::emulate_threads(void)
{
  if (verbose)
    fprintf(stdout,"WEFT INFO: Emulating %d GPU threads "
                   "with %d CPU threads...\n",
                   max_num_threads, thread_pool_size);
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
  pthread_exit(NULL);
}

EmulateTask::EmulateTask(Thread *t)
  : thread(t)
{
}

void EmulateTask::execute(void)
{
  thread->emulate();
}

int main(int argc, char **argv)
{
  Weft weft(argc, argv);
  weft.verify();
  return 0;
}

