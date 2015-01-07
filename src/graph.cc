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
#include "graph.h"
#include "program.h"
#include "instruction.h"

BarrierInstance::BarrierInstance(BarrierDependenceGraph *g,
                                 int n, int gen)
  : graph(g), name(n), generation(gen),
    base_incoming(-1), base_outgoing(-1)
{
  incoming.resize(graph->max_num_barriers, NULL);
  outgoing.resize(graph->max_num_barriers, NULL);
}

BarrierInstance::BarrierInstance(const BarrierInstance &rhs)
  : graph(NULL), name(0), generation(0) 
{
  assert(false);
}

BarrierInstance::~BarrierInstance(void)
{
}

BarrierInstance& BarrierInstance::operator=(const BarrierInstance &rhs)
{
  assert(false);
  return *this;
}

void BarrierInstance::update_waiting_threads(std::set<Thread*> &waiting_threads)
{
  // Add all the non-arrival participants to the set
  for (std::vector<WeftBarrier*>::const_iterator it = participants.begin();
        it != participants.end(); it++)
  {
    if ((*it)->is_arrive())
      continue;
    waiting_threads.insert((*it)->thread);
  }
}

bool BarrierInstance::intersects_with(const std::set<Thread*> &waiting_threads)
{
  for (std::vector<WeftBarrier*>::const_iterator it = participants.begin();
        it != participants.end(); it++)
  {
    if (waiting_threads.find((*it)->thread) != waiting_threads.end())
      return true;
  }
  return false;
}

bool BarrierInstance::happens_after(BarrierInstance *other)
{
  return other->happens_before(participants);
}

bool BarrierInstance::happens_before(
                const std::vector<WeftBarrier*> &other_participants)
{
  // Go through all the other participants and see if we can 
  // find a commong thread
  for (std::vector<WeftBarrier*>::const_iterator it = 
        other_participants.begin(); it != other_participants.end(); it++)
  {
    std::map<Thread*,WeftBarrier*>::const_iterator finder = 
      syncs_only.find((*it)->thread);
    if (finder != syncs_only.end())
    {
      // Found a common thread, see if they are both syncs or a
      // sync before an arrive
      WeftInstruction *ours = finder->second;
      WeftInstruction *theirs = (*it);
      assert(ours != theirs);
      // The only way to get a happens before relationshiop is if
      // their instruction comes after ours
      if (ours->thread_line_number < theirs->thread_line_number)
        return true;
      else // A little sanity check
        assert(theirs->is_arrive());
    }
  }
  return false;
}

void BarrierInstance::add_participant(WeftBarrier *participant, bool sync)
{
  participants.push_back(participant);
  if (sync)
  {
    assert(syncs_only.find(participant->thread) == syncs_only.end());
    syncs_only[participant->thread] = participant;
  }
  participant->set_instance(this);
}

bool BarrierInstance::has_next(BarrierInstance *other)
{
  BarrierInstance *local = outgoing[other->name];
  if (local == NULL)
    return false;
  if (local->generation < other->generation)
    return true;
  return false;
}

bool BarrierInstance::has_previous(BarrierInstance *other)
{
  BarrierInstance *local = incoming[other->name];
  if (local == NULL)
    return false;
  if (local->generation > other->generation)
    return true;
  return false;
}

void BarrierInstance::add_incoming(BarrierInstance *other)
{
  assert(other != NULL);
  BarrierInstance *&current = incoming[other->name];
  // Keep the latest one before the barrier
  if (current == NULL)
    current = other;
  else if (other->generation > current->generation)
  {
    // Remove ourselves from its outgoing list
    current->remove_outgoing(name, generation);
    current = other;
  }
}

void BarrierInstance::add_outgoing(BarrierInstance *other)
{
  assert(other != NULL);
  BarrierInstance *&current = outgoing[other->name];
  // Keep the earliest after the barrier
  if (current == NULL)
    current = other;
  else if (other->generation < current->generation)
  {
    // Remove ourselves from the its incoming list
    current->remove_incoming(name, generation);
    current = other;
  }
}

void BarrierInstance::remove_incoming(int other_name, int gen)
{
  assert(incoming[other_name] != NULL);
  if (incoming[other_name]->generation == gen)
    incoming[other_name] = NULL;
}

void BarrierInstance::remove_outgoing(int other_name, int gen)
{
  assert(outgoing[other_name] != NULL);
  if (outgoing[other_name]->generation == gen)
    outgoing[other_name] = NULL;
}

void BarrierInstance::initialize_pending_counts(void)
{
  // If we haven't computed it yet, do it now
  if (base_incoming == -1)
  {
    base_incoming = 0;
    for (std::vector<BarrierInstance*>::const_iterator it = 
          incoming.begin(); it != incoming.end(); it++)
    {
      if ((*it) != NULL)
        base_incoming++;
    }
  }
  if (base_outgoing == -1)
  {
    base_outgoing = 0;
    for (std::vector<BarrierInstance*>::const_iterator it = 
          outgoing.begin(); it != outgoing.end(); it++)
    {
      if ((*it) != NULL)
        base_outgoing++;
    }
  }
  // +1 because we do an initial check
  pending_incoming = base_incoming + 1;
  pending_outgoing = base_outgoing + 1;
}

template<typename T>
void BarrierInstance::launch_if_ready(Weft *weft, bool forward)
{
  if (forward)
  {
    int remaining = __sync_sub_and_fetch(&pending_incoming, 1);
    if (remaining == 0)
      weft->enqueue_task(new T(this, weft, true/*forward*/));
  }
  else
  {
    int remaining = __sync_sub_and_fetch(&pending_outgoing, 1);
    if (remaining == 0)
      weft->enqueue_task(new T(this, weft, false/*forward*/));
  }
}

template<typename T>
void BarrierInstance::notify_dependences(Weft *weft, bool forward)
{
  if (forward)
  {
    for (std::vector<BarrierInstance*>::const_iterator it = 
          outgoing.begin(); it != outgoing.end(); it++)
    {
      if ((*it) == NULL)
        continue;
      (*it)->launch_if_ready<T>(weft, true/*forward*/);
    }
  }
  else
  {
    for (std::vector<BarrierInstance*>::const_iterator it = 
          incoming.begin(); it != incoming.end(); it++)
    {
      if ((*it) == NULL)
        continue;
      (*it)->launch_if_ready<T>(weft, false/*forward*/);
    }
  }
}

void BarrierInstance::compute_reachability(Weft *weft, bool forward)
{
  // Do the computation
  if (forward)
  {
    // Initialize our vector with our incoming
    latest_incoming = incoming;
    // Then update the incoming based on what all our incoming can reach
    for (std::vector<BarrierInstance*>::const_iterator it = 
          incoming.begin(); it != incoming.end(); it++)
    {
      if ((*it) == NULL)
        continue;
      (*it)->update_latest_incoming(latest_incoming);
    }
  }
  else
  {
    for (std::vector<BarrierInstance*>::const_iterator it = 
          outgoing.begin(); it != outgoing.end(); it++)
    {
      if ((*it) == NULL)
        continue;
      (*it)->update_earliest_outgoing(earliest_outgoing);
    }
  }
  // Then update all our dependences
  notify_dependences<ReachabilityTask>(weft, forward);
}

void BarrierInstance::compute_transitivity(Weft *weft, bool forward)
{
  // Do the computation
  // Initialize our line numbers
  // Then do the transitive update
  if (forward)
  {
    latest_before.resize(graph->weft->thread_count(), -1);
    for (std::vector<WeftBarrier*>::const_iterator it = 
          participants.begin(); it != participants.end(); it++)
    {
      latest_before[(*it)->thread->thread_id] = (*it)->thread_line_number;
    }
    // Now check all our latest incoming barriers for transitive cases
    for (std::vector<BarrierInstance*>::const_iterator it = 
          latest_incoming.begin(); it != latest_incoming.end(); it++)
    {
      if ((*it) == NULL)
        continue;
      (*it)->update_latest_before(latest_before);
    }
  }
  else
  {
    earliest_after.resize(graph->weft->thread_count(), -1);
    for (std::vector<WeftBarrier*>::const_iterator it = 
          participants.begin(); it != participants.end(); it++)
    {
      // We can't count arrives as providing a happens-after relationship
      if ((*it)->is_arrive())
        continue;
      earliest_after[(*it)->thread->thread_id] = (*it)->thread_line_number;
    }
    // Now check all our earliest before barriers for transitive cases
    for (std::vector<BarrierInstance*>::const_iterator it = 
          earliest_outgoing.begin(); it != earliest_outgoing.end(); it++)
    {
      (*it)->update_earliest_after(earliest_after);
    }
  }
  // Then update all our dependences
  notify_dependences<TransitiveTask>(weft, forward);
}

void BarrierInstance::update_latest_incoming(
                              std::vector<BarrierInstance*> &other)
{
  unsigned idx = 0;
  for (std::vector<BarrierInstance*>::const_iterator it = 
        latest_incoming.begin(); it != latest_incoming.end(); it++, idx++)
  {
    if ((*it) == NULL)
      continue;
    BarrierInstance *bar = other[idx];
    // If we have no previous barrier or we have a later generation
    // of the same named barrier then we use that
    if ((bar == NULL) || ((*it)->generation > bar->generation))
      other[idx] = (*it);
  }
}

void BarrierInstance::update_earliest_outgoing(
                              std::vector<BarrierInstance*> &other)
{
  unsigned idx = 0;
  for (std::vector<BarrierInstance*>::const_iterator it = 
        earliest_outgoing.begin(); it != earliest_outgoing.end(); it++, idx++)
  {
    if ((*it) == NULL)
      continue;
    BarrierInstance *bar = other[idx];
    if ((bar == NULL) || ((*it)->generation < bar->generation))
      other[idx] = (*it);
  }
}

void BarrierInstance::update_latest_before(std::vector<int> &other)
{
  unsigned idx = 0;
  for (std::vector<int>::const_iterator it = latest_before.begin();
        it != latest_before.end(); it++, idx++)
  {
    if ((*it) == -1)
      continue;
    int latest = other[idx];
    if ((latest == -1) || ((*it) > latest))
      other[idx] = (*it);
  }
}

void BarrierInstance::update_earliest_after(std::vector<int> &other)
{
  unsigned idx = 0;
  for (std::vector<int>::const_iterator it = earliest_after.begin();
        it != earliest_after.end(); it++, idx++)
  {
    if ((*it) == -1)
      continue;
    int earliest = other[idx];
    if ((earliest == -1) || ((*it) < earliest))
      other[idx] = (*it);
  }
}

void BarrierInstance::traverse_forward(std::deque<BarrierInstance*> &queue,
                                       std::set<BarrierInstance*> &visited)
{
  for (std::vector<BarrierInstance*>::const_iterator it = outgoing.begin();
        it != outgoing.end(); it++)
  {
    if ((*it) == NULL)
        continue;
    std::set<BarrierInstance*>::const_iterator finder = visited.find(*it);
    if (finder == visited.end())
    {
      queue.push_back(*it);
      visited.insert(*it);
    }
  }
}

void BarrierDependenceGraph::PreceedingBarriers::find_preceeding(
                                              BarrierInstance *next)
{
  if (arrival_threads.empty())
    return;
  bool intersect = next->intersects_with(arrival_threads);
  if (intersect)
  {
    // Walk backwards and see if we find an intersection
    for (std::deque<BarrierInstance*>::const_iterator it = 
          previous.begin(); it != previous.end(); it++)
    {
      // See if they intersect on any threads
      if (next->happens_after(*it))
      {
        // Check to make sure we can add this edge for
        // both of them. If they already have an 
        // earlier/later version of the same physical
        // barrier then there is no need to add the edge.
        if (!(*it)->has_next(next) &&
            !next->has_previous(*it))
        {
          (*it)->add_outgoing(next);
          next->add_incoming(*it);
        }
        // We found one so we are done
        return;
      }
    }
  }
}

void BarrierDependenceGraph::PreceedingBarriers::add_instance(
                                              BarrierInstance *next)
{
  previous.push_front(next);
  // Keep track of any threads that have touched this 
  // physical named barrier as a fast way of doing
  // interference testing
  next->update_waiting_threads(arrival_threads);
}

BarrierDependenceGraph::BarrierDependenceGraph(Weft *w)
  : weft(w), max_num_barriers(w->barrier_upper_bound())
{
  barrier_instances.resize(max_num_barriers);
  PTHREAD_SAFE_CALL( pthread_mutex_init(&validation_mutex, NULL) );
}

BarrierDependenceGraph::BarrierDependenceGraph(
                        const BarrierDependenceGraph &rhs)
  : weft(NULL), max_num_barriers(0)
{
  assert(false);
}

BarrierDependenceGraph::~BarrierDependenceGraph(void)
{
  barrier_instances.clear();
  for (std::deque<BarrierInstance*>::iterator it = 
        all_barriers.begin(); it != all_barriers.end(); it++)
  {
    delete (*it);
  }
  all_barriers.clear();
  PTHREAD_SAFE_CALL( pthread_mutex_destroy(&validation_mutex) );
}

BarrierDependenceGraph& BarrierDependenceGraph::operator=(
                            const BarrierDependenceGraph &rhs)
{
  assert(false);
  return *this;
}

void BarrierDependenceGraph::construct_graph(
                            const std::vector<Thread*> &threads)
{
  std::vector<int> program_counters(threads.size(), 0); 
  std::vector<PendingState> pending_arrives(max_num_barriers);
  std::vector<PreceedingBarriers> preceeding_barriers(max_num_barriers);
  bool has_deadlock = false;
  while (true)
  {
    if (remove_complete_barriers(program_counters, pending_arrives,
                             preceeding_barriers, threads))
      continue;
    if (are_empty(program_counters, threads))
      break;
    else if (!advance_program_counters(program_counters, 
                                       pending_arrives, threads))
    {
      has_deadlock = true;
      break;
    }
  }
  if (has_deadlock)
  {
    if (weft->print_verbose())
    {
      report_state(program_counters, threads, pending_arrives);
      weft->report_error(WEFT_ERROR_DEADLOCK, "DEADLOCK DETECTED! "
                      "(thread and barrier state reported above)");
    }
    else
      weft->report_error(WEFT_ERROR_DEADLOCK, "DEADLOCK DETECTED! "
          "(run in verbose mode to see thread and barrier state)");
  }
  else
    fprintf(stdout,"WEFT INFO: No deadlocks detected!\n");
  if (weft->print_verbose())
    fprintf(stdout,"WEFT INFO: Total barrier instances: %ld\n",
            all_barriers.size());
}

int BarrierDependenceGraph::count_validation_tasks(void)
{
  int result = 0;
  for (std::vector<std::deque<BarrierInstance*> >::iterator it = 
        barrier_instances.begin(); it != barrier_instances.end(); it++)
  {
    if (it->empty())
      continue;
    result += (it->size() - 1);
  }
  return result;
}

void BarrierDependenceGraph::enqueue_validation_tasks(void)
{
  unsigned name = 0;
  for (std::vector<std::deque<BarrierInstance*> >::iterator it = 
        barrier_instances.begin(); it != barrier_instances.end(); it++, name++)
  {
    if (it->empty())
      continue;
    std::deque<BarrierInstance*> &local = *it; 
    for (unsigned idx = 0; idx < (local.size()-1); idx++)
    {
      ValidationTask *task = new ValidationTask(this, name, idx);
      weft->enqueue_task(task);
    }
  }
}

void BarrierDependenceGraph::check_for_validation_errors(void)
{
  // No need to hold the lock here since we are done with the
  // threadpool when we invokee this method
  if (!failed_validations.empty())
  {
    fprintf(stderr, "WEFT INFO: BARRIERS NOT PROPERLY RECYCLED!\n");
    for (std::vector<std::pair<int,int> >::const_iterator it = 
          failed_validations.begin(); it != failed_validations.end(); it++)
    {
      fprintf(stderr,"  Unable to find happens-before relationship "
                     "between generations %d and %d of named barrier %d\n",
                     it->second, it->second+1, it->first);
    }
    char buffer[1024];
    snprintf(buffer, 1023, "Unable to find happens before relationships "
                           "for %ld different named barrier generations",
                           failed_validations.size());
    weft->report_error(WEFT_ERROR_GRAPH_VALIDATION, buffer);
  }
  else
    fprintf(stdout,"WEFT INFO: Barriers properly recycled!\n");
}

void BarrierDependenceGraph::validate_barrier(int name, int generation)
{
  // Do a breadth-first search to get from generation to generation+1
  assert(name < barrier_instances.size());
  std::deque<BarrierInstance*> &named_barrier = barrier_instances[name];
  assert((generation+1) < named_barrier.size());
  BFSSearch bfs(named_barrier[generation], named_barrier[generation+1]);
  bool found = bfs.execute();
  if (!found)
  {
    PTHREAD_SAFE_CALL( pthread_mutex_lock(&validation_mutex) );
    failed_validations.push_back(std::pair<int,int>(name, generation));
    PTHREAD_SAFE_CALL( pthread_mutex_unlock(&validation_mutex) );
  }
}

int BarrierDependenceGraph::count_total_barriers(void)
{
  return all_barriers.size();
}

void BarrierDependenceGraph::enqueue_reachability_tasks(void)
{
  // We leverage our knowledge of graph topology to compute
  // this intelligently. We start off launching tasks only
  // for barriers with no incoming edges for happens-after and
  // no outgoing edges for happens-before. Each task execution
  // then launches any tasks which have had all its dependences
  // satisfied. This keeps us from having to do a bunch of 
  // expensive BFS searches over the entire graph.

  // First initialize the pending counts for all barriers
  // before we start launching any tasks
  initialize_pending_counts();
  for (std::deque<BarrierInstance*>::const_iterator it = 
        all_barriers.begin(); it != all_barriers.end(); it++)
  {
    (*it)->launch_if_ready<ReachabilityTask>(weft, true/*forward*/); 
    (*it)->launch_if_ready<ReachabilityTask>(weft, false/*forward*/);
  }
}

void BarrierDependenceGraph::enqueue_transitive_happens_tasks(void)
{
  // This operates the same as the reachability tasks to avoid
  // having to see when all the inputs have been updated.
  initialize_pending_counts();
  for (std::deque<BarrierInstance*>::const_iterator it = 
        all_barriers.begin(); it != all_barriers.end(); it++)
  {
    (*it)->launch_if_ready<TransitiveTask>(weft, true/*forward*/); 
    (*it)->launch_if_ready<TransitiveTask>(weft, false/*forward*/);
  }
}

void BarrierDependenceGraph::initialize_pending_counts(void)
{
  for (std::deque<BarrierInstance*>::const_iterator it = 
        all_barriers.begin(); it != all_barriers.end(); it++)
  {
    (*it)->initialize_pending_counts();
  }
}

bool BarrierDependenceGraph::remove_complete_barriers(
                                std::vector<int> &program_counters,
                                std::vector<PendingState> &pending_arrives,
                                std::vector<PreceedingBarriers> &preceeding,
                                const std::vector<Thread*> &threads)
{
  bool removed_barrier = false;
  std::vector<int> barrier_expected(max_num_barriers, -1);
  std::vector<int> barrier_participants(max_num_barriers, -1);
  std::vector<bool> all_arrives(max_num_barriers, true);
  // Initialize the pending state for all barriers
  for (unsigned name = 0; name < max_num_barriers; name++)
  {
    PendingState &state = pending_arrives[name];
    if (state.expected <= 0)
      continue;
    barrier_expected[name] = state.expected;
    barrier_participants[name] = state.arrivals.size();
  }
  // Scan across all the threads and see if we can find any complete
  // barriers, if so pop them off the stack and advance program counters
  unsigned idx = 0;
  for (std::vector<Thread*>::const_iterator it = threads.begin();
        it != threads.end(); it++, idx++)
  {
    WeftInstruction *inst = (*it)->get_instruction(program_counters[idx]);
    // If we're done we can skip this thread
    if (inst == NULL)
      continue;
    // Check to see if a barrier is at the top
    if (inst->is_barrier())
    {
      WeftBarrier *bar = inst->as_barrier(); 
      int name = bar->name;
      assert((name >= 0) && (name < max_num_barriers));
      // See if we've seen this barrier ID before
      if (barrier_expected[name] == -1)
      {
        // If not, set up the pending arrivals
        assert(barrier_participants[name] == -1);
        barrier_expected[name] = bar->count;
        barrier_participants[name] = 1;
      }
      else
      {
        // Otherwise, update the pending counts
        if (barrier_expected[name] != bar->count)
        {
          char buffer[1024];
          snprintf(buffer, 1023, "Different arrival counts of %d and %d "
                                 "possible on barrier %d",
                                 barrier_expected[name], bar->count, name);
          weft->report_error(WEFT_ERROR_ARRIVAL_MISMATCH, buffer);
        }
        barrier_participants[name]++;
      }
      if (bar->is_sync())
        all_arrives[name] = false;
    }
  }
  // Now let's see which barriers are complete
  for (unsigned name = 0; name < max_num_barriers; name++)
  {
    if (barrier_expected[name] == -1)
      continue;
    if (barrier_participants[name] > barrier_expected[name])
    {
      char buffer[1024];
      snprintf(buffer, 1023, "Too many participants (%d) for barrier %d while "
                             "expecting only %d participants", 
                             barrier_participants[name], name,
                             barrier_expected[name]);
      weft->report_error(WEFT_ERROR_TOO_MANY_PARTICIPANTS, buffer);
    }
    if (barrier_participants[name] == barrier_expected[name])
    {
      if (all_arrives[name])
      {
        char buffer[1024];
        snprintf(buffer, 1023, "All arrivals on barrier %d possible", name);
        weft->report_error(WEFT_ERROR_ALL_ARRIVALS, buffer);
      }
      // Mark that we removed a barrier
      removed_barrier = true;
      // We have a complete barrier so let's pop everything off the stacks
      PendingState &state = pending_arrives[name];
      // Create a new barrier instance
      BarrierInstance *bar_inst = 
                        new BarrierInstance(this, name, state.generation);
      barrier_instances[name].push_back(bar_inst);
      all_barriers.push_back(bar_inst);
      idx = 0;
      for (std::vector<Thread*>::const_iterator it = threads.begin();
            it != threads.end(); it++, idx++)
      {
        WeftInstruction *inst = (*it)->get_instruction(program_counters[idx]);
        if (inst == NULL)
          continue;
        if (!inst->is_barrier())
          continue;
        WeftBarrier *bar = inst->as_barrier();  
        if (bar->name != name)
          continue;
        bar_inst->add_participant(bar, bar->is_sync());
        // We can advance the counter since we handled the instruction
        program_counters[idx]++;
      }
      // Also handle the pending arrivals
      if (state.expected != -1)
      {
        for (std::set<BarrierArrive*>::const_iterator it = 
              state.arrivals.begin(); it != state.arrivals.end(); it++)
        {
          bar_inst->add_participant(*it, false/*sync*/);
        }
      }
      // Reet the state
      state.reset();
      // If we made a new barrier, then find all the immediately
      // preceeding instances of barriers on which this barrier
      // is guaranteed to have a happens-before relationship
      for (idx = 0; idx < max_num_barriers; idx++)
        preceeding[idx].find_preceeding(bar_inst);
      // Add this barrier to the preceeding instances for the given name
      preceeding[name].add_instance(bar_inst);
    }
  }
  return removed_barrier;
}

bool BarrierDependenceGraph::are_empty(const std::vector<int> &program_counters,
                                       const std::vector<Thread*> &threads)
{
  for (unsigned idx = 0; idx < threads.size(); idx++)
  {
    size_t size = threads[idx]->get_program_size();
    assert(program_counters[idx] <= size);
    if (program_counters[idx] < size)
      return false;
  }
  return true;
}

bool BarrierDependenceGraph::advance_program_counters(
                                std::vector<int> &program_counters,
                                std::vector<PendingState> &pending_arrives,
                                const std::vector<Thread*> &threads)
{
  bool found = false;
  unsigned idx = 0;
  for (std::vector<Thread*>::const_iterator it = threads.begin();
        it != threads.end(); it++, idx++)
  {
    if (program_counters[idx] == (*it)->get_program_size())
      continue;
    WeftInstruction *inst = (*it)->get_instruction(program_counters[idx]);
    while (!inst->is_barrier() || inst->is_arrive())
    {
      if (inst->is_arrive())
      {
        // If it is an arrival, pop it off and put in the
        // pending arrive data structure
        BarrierArrive *arrive = inst->as_arrive();
        assert((arrive->name >= 0) && (arrive->name < max_num_barriers));
        PendingState &state = pending_arrives[arrive->name];
        if (state.expected == -1)
          state.expected = arrive->count;
        else if (state.expected != arrive->count)
        {
          char buffer[1024];
          snprintf(buffer, 1023, "Different arrival counts of %d and %d "
                                 "possible on barrier %d",
                                 state.expected, arrive->count, arrive->name);
          weft->report_error(WEFT_ERROR_ARRIVAL_MISMATCH, buffer);
        }
        state.arrivals.insert(arrive);
        if (state.expected == state.arrivals.size())
        {
          char buffer[1024];
          snprintf(buffer, 1023, "All arrivals on barrier %d possible",
                                  arrive->name);
          weft->report_error(WEFT_ERROR_ALL_ARRIVALS, buffer);
        }
      }
      // Otherwise it is a shared memory access so we can
      // safely ignore it for now
      found = true;
      program_counters[idx]++;
      inst = (*it)->get_instruction(program_counters[idx]);
      if (inst == NULL)
        break;
    }
  }
  return found;
}

void BarrierDependenceGraph::report_state(
                                const std::vector<int> &program_counters,
                                const std::vector<Thread*> &threads, 
                                const std::vector<PendingState> &pending_arrives)
{
  unsigned idx = 0;
  for (std::vector<Thread*>::const_iterator it = threads.begin(); 
        it != threads.end(); it++, idx++)
  {
    WeftInstruction *inst = (*it)->get_instruction(program_counters[idx]); 
    if (inst != NULL)
    {
      assert(inst->is_sync());
      BarrierSync *sync = inst->as_sync();
      fprintf(stderr,"  Thread %d: Blocked on barrier %d (PTX line %d)\n", 
                      idx, sync->name, sync->instruction->line_number);
    }
    else
      fprintf(stderr,"  Thread %d: Exited\n", idx);
  }
  fprintf(stderr,"\n");
  idx = 0;
  for (std::vector<PendingState>::const_iterator it = pending_arrives.begin();
        it != pending_arrives.end(); it++, idx++)
  {
    if (it->expected > 0)
      fprintf(stderr,"  Barrier %d (generation %d) has observed "
                     "%ld arrivals for %d expected participants\n", 
                     idx, it->generation, it->arrivals.size(), it->expected); 
    else
      fprintf(stderr,"  Barrier %d (generation %d) has observed "
                     "%ld arrivals for an unknown number of participants\n", 
                     idx, it->generation, it->arrivals.size());
  }
  fprintf(stderr,"\n");
}

ValidationTask::ValidationTask(BarrierDependenceGraph *g, int n, int gen)
  : WeftTask(), graph(g), name(n), generation(gen)
{
}

void ValidationTask::execute(void)
{
  graph->validate_barrier(name, generation);
}

BFSSearch::BFSSearch(BarrierInstance *src, BarrierInstance *tar)
  : source(src), target(tar)
{
  queue.push_back(src);
  visited.insert(src);
}

bool BFSSearch::execute(void)
{
  while (!queue.empty())
  {
    BarrierInstance *next = queue.front();
    queue.pop_front();
    if (next == target)
      return true;
    // If this ever happens, that is really bad
    if ((next->name == target->name) &&
        (next->generation > target->generation))
      return false;
    next->traverse_forward(queue, visited);
  }
  return false;
}

ReachabilityTask::ReachabilityTask(BarrierInstance *inst, Weft *w, bool f)
  : WeftTask(), instance(inst), weft(w), forward(f)
{
}

void ReachabilityTask::execute(void)
{
  instance->compute_reachability(weft, forward);
}

TransitiveTask::TransitiveTask(BarrierInstance *inst, Weft *w, bool f)
  : WeftTask(), instance(inst), weft(w), forward(f)
{
}

void TransitiveTask::execute(void)
{
  instance->compute_transitivity(weft, forward);
}

