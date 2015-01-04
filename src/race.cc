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
    (*it)->get_instance()->update_latest_before(happens_before);
  }
  for (std::vector<WeftBarrier*>::const_iterator it = 
        earliest_after.begin(); it != earliest_after.end(); it++)
  {
    if ((*it) == NULL)
      continue;
    (*it)->get_instance()->update_earliest_after(happens_after);
  }
}

Address::Address(const int addr)
  : address(addr)
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
    address = new Address(access->address);
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

}

RaceCheckTask::RaceCheckTask(Address *addr)
  : address(addr)
{
}

void RaceCheckTask::execute(void)
{
  address->perform_race_tests();
}

