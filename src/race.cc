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
#include "race.h"
#include "graph.h"
#include "program.h"
#include "instruction.h"

Happens::Happens(int total_threads)
  : initialized(false)
{
  happens_before.resize(total_threads, -1);
  happens_after.resize(total_threads, -1);
}

void Happens::update_barriers_before(const std::vector<WeftBarrier*> &before)
{
  assert(latest_before.empty());
  latest_before = before;
}

void Happens::update_barriers_after(const std::vector<WeftBarrier*> &after)
{
  assert(earliest_after.empty());
  earliest_after = after;
}

void Happens::update_happens_relationships(void)
{
  for (std::vector<WeftBarrier*>::const_iterator it = 
        latest_before.begin(); it != latest_before.end(); it++)
  {
    if ((*it) == NULL)
      continue;
    (*it)->get_instance()->update_latest_before(happens_after);
  }
  for (std::vector<WeftBarrier*>::const_iterator it = 
        earliest_after.begin(); it != earliest_after.end(); it++)
  {
    if ((*it) == NULL)
      continue;
    (*it)->get_instance()->update_earliest_after(happens_before);
  }
}

bool Happens::has_happens(int thread, int line_number)
{
  if (happens_before[thread] <= line_number)
    return true;
  if (happens_after[thread] >= line_number)
    return true;
  return false;
}

Address::Address(const int addr, SharedMemory *mem)
  : address(addr), memory(mem), total_races(0)
{
  PTHREAD_SAFE_CALL( pthread_mutex_init(&address_lock,NULL) );
}

Address::~Address(void)
{
  PTHREAD_SAFE_CALL( pthread_mutex_destroy(&address_lock) );
}

void Address::add_access(WeftAccess *access)
{
  PTHREAD_SAFE_CALL( pthread_mutex_lock(&address_lock) );
  accesses.push_back(access);
  PTHREAD_SAFE_CALL( pthread_mutex_unlock(&address_lock) );
}

void Address::perform_race_tests(void)
{
  if (memory->weft->assume_warp_synchronous())
  {
    for (unsigned idx1 = 0; idx1 < accesses.size(); idx1++)
    {
      WeftAccess *first = accesses[idx1];
      if (first->is_read())
      {
        for (unsigned idx2 = idx1+1; idx2 < accesses.size(); idx2++)
        {
          WeftAccess *second = accesses[idx2]; 
          // Check for both reads
          if (second->is_read())
            continue;
          // Check for warp-synchronous
          if (first->is_warp_synchronous(second))
            continue;
          if (!first->has_happens_relationship(second))
            record_race(first, second);
        }
      }
      else
      {
        for (unsigned idx2 = idx1+1; idx2 < accesses.size(); idx2++)
        {
          WeftAccess *second = accesses[idx2];
          // Check for warp-synchronous
          if (first->is_warp_synchronous(second))
            continue;
          if (!first->has_happens_relationship(second))
            record_race(first, second);
        }
      }
    }
  }
  else
  {
    // For every pair of addresses, check to see if we can 
    // establish a happens before or a happens after relationship
    for (unsigned idx1 = 0; idx1 < accesses.size(); idx1++)
    {
      WeftAccess *first = accesses[idx1];
      if (first->is_read())
      {
        for (unsigned idx2 = idx1+1; idx2 < accesses.size(); idx2++)
        {
          WeftAccess *second = accesses[idx2]; 
          // Check for both reads
          if (second->is_read())
            continue;
          if (!first->has_happens_relationship(second))
            record_race(first, second);
        }
      }
      else
      {
        for (unsigned idx2 = idx1+1; idx2 < accesses.size(); idx2++)
        {
          WeftAccess *second = accesses[idx2];
          if (!first->has_happens_relationship(second))
            record_race(first, second);
        }
      }
    }
  }
}

void Address::record_race(WeftAccess *one, WeftAccess *two)
{
  // Alternative race reporting
  //printf("Race between threads %d and %d on instructions "
  //       "%d and %d (PTX %d and %d)\n",
  //       one->thread->thread_id, two->thread->thread_id,
  //       one->thread_line_number, two->thread_line_number,
  //       one->instruction->line_number, two->instruction->line_number);
  total_races++;
  // Save the races based on the PTX instructions
  int ptx_one = one->instruction->line_number;
  int ptx_two = two->instruction->line_number;
  if (ptx_one <= ptx_two)
  {
    std::pair<PTXInstruction*,PTXInstruction*> 
                            key(one->instruction, two->instruction);
    if (one->thread->thread_id <= two->thread->thread_id)
      ptx_races[key].insert(
          std::pair<Thread*,Thread*>(one->thread, two->thread));
    else
      ptx_races[key].insert(
          std::pair<Thread*,Thread*>(two->thread, one->thread));
  }
  else
  {
    std::pair<PTXInstruction*,PTXInstruction*> 
                            key(two->instruction, one->instruction);
    if (one->thread->thread_id <= two->thread->thread_id)
      ptx_races[key].insert(
          std::pair<Thread*,Thread*>(one->thread, two->thread));
    else
      ptx_races[key].insert(
          std::pair<Thread*,Thread*>(two->thread, one->thread));
  }
}

int Address::report_races(std::map<
            std::pair<PTXInstruction*,PTXInstruction*>,size_t> &all_races)
{
  if (total_races > 0)
  { 
    if (memory->weft->print_detail())
    {
      fprintf(stderr,"WEFT INFO: Found %d races on adress %d!\n",
                      total_races, address);
      for (std::map<std::pair<PTXInstruction*,PTXInstruction*>,std::set<
                    std::pair<Thread*,Thread*> > >::const_iterator it = 
            ptx_races.begin(); it != ptx_races.end(); it++)
      {
        PTXInstruction *one = it->first.first;
        PTXInstruction *two = it->first.second;
        if (one->source_file != NULL)
        {
          assert(two->source_file != NULL);
          if (one == two)
            fprintf(stderr,"\tThere are %ld races between different threads "
                  "on line %d of %s with address %d\n", it->second.size(),
                  one->source_line_number, one->source_file, address);
          else
            fprintf(stderr,"\tThere are %ld races between line %d of %s "
                    " and line %d of %s with address %d\n", it->second.size(),
                    one->source_line_number, one->source_file,
                    two->source_line_number, two->source_file, address);
        }
        else
        {
          assert(two->source_file == NULL);
          if (one == two)
            fprintf(stderr,"\tThere are %ld races between different threads "
                   "on PTX line %d with address %d\n", it->second.size(),
                   one->line_number, address);
          else
            fprintf(stderr,"\tThere are %ld races between PTX line %d "
                    " and PTX line %d with address %d\n", it->second.size(),
                    one->line_number, two->line_number, address);
        }
        const std::set<std::pair<Thread*,Thread*> > &threads = it->second;
        for (std::set<std::pair<Thread*,Thread*> >::const_iterator 
              thread_it = threads.begin(); 
              thread_it != threads.end(); thread_it++)
        {
          Thread *first = thread_it->first;
          Thread *second = thread_it->second;
          fprintf(stderr,"\t\t... between thread (%d,%d,%d) and (%d,%d,%d)\n",
                  first->tid_x, first->tid_y, first->tid_z,
                  second->tid_x, second->tid_y, second->tid_z);
        }
      }
    }
    else
    {
      for (std::map<std::pair<PTXInstruction*,PTXInstruction*>,
                    std::set<std::pair<Thread*,Thread*> > >::const_iterator
            it = ptx_races.begin(); it != ptx_races.end(); it++)
      {
        std::map<std::pair<PTXInstruction*,PTXInstruction*>,size_t>::iterator
          finder = all_races.find(it->first);
        if (finder == all_races.end())
          all_races[it->first] = it->second.size();
        else
          finder->second += it->second.size();
      }
    }
  }
  return total_races;
}

size_t Address::count_race_tests(void)
{
  size_t num_accesses = accesses.size();
  // OLA's equality
  // 1 + 2 + 3 + ... + n-1 = (n-1)*n/2
  return ((num_accesses * (num_accesses-1))/2);
}

SharedMemory::SharedMemory(Weft *w)
  : weft(w)
{
  PTHREAD_SAFE_CALL( pthread_mutex_init(&memory_lock,NULL) );
}

SharedMemory::~SharedMemory(void)
{
  for (std::map<int,Address*>::iterator it = addresses.begin();
        it != addresses.end(); it++)
  {
    delete it->second;
  }
  addresses.clear();
  PTHREAD_SAFE_CALL( pthread_mutex_destroy(&memory_lock) );
}

void SharedMemory::update_accesses(WeftAccess *access)
{
  Address *address;
  // These lookups need to be thread safe
  PTHREAD_SAFE_CALL( pthread_mutex_lock(&memory_lock) );
  std::map<int,Address*>::const_iterator finder = 
    addresses.find(access->address);
  if (finder == addresses.end())
  {
    address = new Address(access->address, this);
    addresses[access->address] = address;
  }
  else
    address = finder->second;
  PTHREAD_SAFE_CALL( pthread_mutex_unlock(&memory_lock) );
  address->add_access(access);
}

int SharedMemory::count_addresses(void) const
{
  return addresses.size();
}

void SharedMemory::enqueue_race_checks(void)
{
  for (std::map<int,Address*>::const_iterator it = addresses.begin();
        it != addresses.end(); it++)
  {
    weft->enqueue_task(new RaceCheckTask(it->second));
  }
}

void SharedMemory::check_for_races(void)
{
  int total_races = 0;
  std::map<std::pair<PTXInstruction*,PTXInstruction*>,size_t> all_races;
  for (std::map<int,Address*>::const_iterator it = 
        addresses.begin(); it != addresses.end(); it++)
  {
    total_races += it->second->report_races(all_races);
  }
  if (total_races > 0)
  {
    if (!weft->print_detail())
    {
      for (std::map<std::pair<PTXInstruction*,PTXInstruction*>,size_t>::const_iterator 
            it = all_races.begin(); it != all_races.end(); it++)
      {
        PTXInstruction *one = it->first.first;
        PTXInstruction *two = it->first.second;
        if (one->source_file != NULL)
        {
          assert(two->source_file != NULL);
          if (one == two)
            fprintf(stderr,"\tFound races between %ld pairs of "
                           "threads on line %d of %s\n", it->second,
                           one->source_line_number, one->source_file);
          else
            fprintf(stderr,"\tFound races between %ld pairs of threads "
                           "on line %d of %s and line %d of %s\n", it->second,
                           one->source_line_number, one->source_file,
                           two->source_line_number, two->source_file);
        }
        else
        {
          assert(two->source_file == NULL);
          if (one == two)
            fprintf(stderr,"\tFound races between %ld pairs of "
                           "threads on PTX line number %d\n",
                           it->second, one->line_number);
          else
            fprintf(stderr,"\tFound races between %ld pairs of threads on "
                           "PTX line %d and PTX line %d\n", it->second,
                           one->line_number, two->line_number);
        }
      }
      fprintf(stderr,"WEFT INFO: Found %d total races!\n"
                     "           Run with '-d' flag to see detailed per-thread "
                     "and per-address races\n", total_races);
    }
    else
      fprintf(stderr,"WEFT INFO: Found %d total races!\n", total_races);
    fprintf(stderr,"WEFT INFO: RACES DETECTED!\n");
  }
  else
    fprintf(stdout,"WEFT INFO: No races detected!\n");
}

size_t SharedMemory::count_race_tests(void)
{
  size_t result = 0;
  for (std::map<int,Address*>::const_iterator it = addresses.begin();
        it != addresses.end(); it++)
  {
    result += it->second->count_race_tests();
  }
  return result;
}

RaceCheckTask::RaceCheckTask(Address *addr)
  : address(addr)
{
}

void RaceCheckTask::execute(void)
{
  address->perform_race_tests();
}

