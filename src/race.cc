
#include "weft.h"
#include "race.h"

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

