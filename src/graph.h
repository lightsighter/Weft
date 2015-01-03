
#ifndef __BARRIER_DEPENDENCE_GRAPH_H__
#define __BARRIER_DEPENDENCE_GRAPH_H__

#include <set>
#include <map>
#include <deque>
#include <vector>
#include <pthread.h>

class Weft;
class Thread;
class WeftBarrier;
class BarrierArrive;
class WeftInstruction;
class BarrierDependenceGraph;

class BarrierInstance {
public:
  BarrierInstance(BarrierDependenceGraph *graph, int name, int generation);
  BarrierInstance(const BarrierInstance &rhs);
  ~BarrierInstance(void);
public:
  BarrierInstance& operator=(const BarrierInstance &rhs);
public:
  void update_waiting_threads(std::set<Thread*> &waiting_threads);
  bool intersects_with(const std::set<Thread*> &waiting_threads);
  bool happens_after(BarrierInstance *other);
  bool happens_before(const std::vector<WeftBarrier*> &other_participants);
public:
  void add_participant(WeftBarrier *participant, bool sync);
  void add_incoming(BarrierInstance *other);
  void add_outgoing(BarrierInstance *other);
public:
  void traverse_forward(std::deque<BarrierInstance*> &queue,
                        std::set<BarrierInstance*> &visited);
public:
  BarrierDependenceGraph *const graph;
  const int name;
  const int generation;
protected:
  std::vector<WeftBarrier*> participants;
  // Helpful for constructing barrier dependence graph
  std::map<Thread*,WeftBarrier*> syncs_only;
protected:
  std::vector<BarrierInstance*> incoming;
  std::vector<BarrierInstance*> outgoing;
};

class BarrierDependenceGraph {
private:
  struct PendingState {
  public:
    PendingState(void)
      : expected(-1), generation(0) { }
  public:
    inline void reset(void) {
      expected = -1;
      generation++;
      arrivals.clear();
    }
  public:
    int expected;
    int generation;
    std::set<BarrierArrive*> arrivals;
  };
  struct PreceedingBarriers {
  public:
    PreceedingBarriers(void) { }
  public:
    void find_preceeding(BarrierInstance *bar);
    void add_instance(BarrierInstance *bar);
  public:
    // This is an upper bound on all arrivals
    std::set<Thread*> arrival_threads;
    std::deque<BarrierInstance*> previous;
  };
public:
  BarrierDependenceGraph(Weft *weft);
  BarrierDependenceGraph(const BarrierDependenceGraph &rhs);
  ~BarrierDependenceGraph(void);
public:
  BarrierDependenceGraph& operator=(const BarrierDependenceGraph &rhs);
public:
  void construct_graph(const std::vector<Thread*> &threads);
  int count_validation_tasks(void);
  void enqueue_validation_tasks(void);
  void check_for_validation_errors(void);
  void validate_barrier(int name, int generation);
protected:
  bool remove_complete_barriers(std::vector<int> &program_counters,
                                std::vector<PendingState> &pending_arrives,
                                std::vector<PreceedingBarriers> &preceeding,
                                const std::vector<Thread*> &threads);
  bool are_empty(const std::vector<int> &program_counters,
                 const std::vector<Thread*> &threads);
  bool advance_program_counters(std::vector<int> &program_counters,
                                std::vector<PendingState> &pending_arrives,
                                const std::vector<Thread*> &threads);
  void report_state(const std::vector<int> &program_counters,
                    const std::vector<Thread*> &threads,
                    const std::vector<PendingState> &pending_arrives);
public:
  Weft *const weft;
  const int max_num_barriers;
protected:
  std::vector<std::deque<BarrierInstance*> > barrier_instances;
protected:
  pthread_mutex_t validation_mutex;
  std::vector<std::pair<int/*name*/,int/*gen*/> > failed_validations;
};

class BFS {
public:
  BFS(BarrierInstance *source, BarrierInstance *target);
  BFS(const BFS &rhs) : source(NULL), target(NULL) { assert(false); }
  ~BFS(void) { }
public:
  BFS& operator=(const BFS &rhs) { assert(false); return *this; }
public:
  bool execute(void);
public:
  BarrierInstance *const source;
  BarrierInstance *const target;
protected:
  std::deque<BarrierInstance*> queue;
  std::set<BarrierInstance*> visited;
};

#endif // __BARRIER_DEPENDENCE_GRAPH_H__
