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
    ptx_races.insert(std::pair<int,int>(ptx_one, ptx_two));
  else
    ptx_races.insert(std::pair<int,int>(ptx_two, ptx_one));
}

int Address::report_races(void)
{
  if ((total_races > 0) && memory->weft->print_verbose())
  {
    fprintf(stderr,"WEFT INFO: Found %d races on adress %d!\n",
                    total_races, address);
    for (std::set<std::pair<int,int> >::const_iterator it = 
          ptx_races.begin(); it != ptx_races.end(); it++)
    {
      fprintf(stderr,"WEFT INFO: Race on address %d between "
                      "PTX instructions on lines %d and %d\n",
                      address, it->first, it->second);
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
  for (std::map<int,Address*>::const_iterator it = 
        addresses.begin(); it != addresses.end(); it++)
  {
    total_races += it->second->report_races();
  }
  if (total_races > 0)
  {
    fprintf(stderr,"WEFT INFO: RACES DETECTED!\n");
    if (weft->print_verbose())
      fprintf(stderr,"WEFT INFO: Found %d total races!\n", total_races);
    else
      fprintf(stderr,"WEFT INFO: Found %d total races!\n"
                     "           Run in verbose mode to see line numbers\n",
                      total_races);
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

