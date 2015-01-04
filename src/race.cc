
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
    (*it)->get_instance()->update_latest_before(happens_before);
  }
  for (std::vector<WeftBarrier*>::const_iterator it = 
        earliest_after.begin(); it != earliest_after.end(); it++)
  {
    (*it)->get_instance()->update_earliest_after(happens_after);
  }
}

