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
  bool has_next(BarrierInstance *other);
  bool has_previous(BarrierInstance *other);
  void add_incoming(BarrierInstance *other);
  void add_outgoing(BarrierInstance *other);
  void remove_incoming(int name, int gen);
  void remove_outgoing(int name, int gen);
public:
  void initialize_pending_counts(void);
  template<typename T>
  void launch_if_ready(Weft *weft, bool forward);
  template<typename T>
  void notify_dependences(Weft *weft, bool forward);
  void compute_reachability(Weft *weft, bool forward);
  void compute_transitivity(Weft *weft, bool forward);
  void update_latest_incoming(std::vector<BarrierInstance*> &other);
  void update_earliest_outgoing(std::vector<BarrierInstance*> &other);
  void update_latest_before(std::vector<int> &other);
  void update_earliest_after(std::vector<int> &other);
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
protected:
  std::vector<BarrierInstance*> latest_incoming;
  std::vector<BarrierInstance*> earliest_outgoing;
protected:
  std::vector<int> latest_before;
  std::vector<int> earliest_after;
protected:
  int base_incoming;
  int base_outgoing;
  int pending_incoming;
  int pending_outgoing;
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
public:
  int count_total_barriers(void);
  void enqueue_reachability_tasks(void);
  void enqueue_transitive_happens_tasks(void);
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
protected:
  void initialize_pending_counts(void);
public:
  Weft *const weft;
  const int max_num_barriers;
protected:
  std::vector<std::deque<BarrierInstance*> > barrier_instances;
  // A summary of all barriers in one place
  std::deque<BarrierInstance*>               all_barriers;
protected:
  pthread_mutex_t validation_mutex;
  std::vector<std::pair<int/*name*/,int/*gen*/> > failed_validations;
};

class BFSSearch {
public:
  BFSSearch(BarrierInstance *source, BarrierInstance *target);
  BFSSearch(const BFSSearch &rhs) : source(NULL), target(NULL) { assert(false); }
  ~BFSSearch(void) { }
public:
  BFSSearch& operator=(const BFSSearch &rhs) { assert(false); return *this; }
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
