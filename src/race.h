
#ifndef __RACE_H__
#define __RACE_H__

#include <vector>
#include <cassert>

class WeftBarrier;

class Happens {
public:
  Happens(int total_threads);
  Happens(const Happens &rhs) { assert(false); }
  ~Happens(void) { }
public:
  Happens& operator=(const Happens &rhs) { assert(false); return *this; }
public:
  void update_barriers_before(const std::vector<WeftBarrier*> &before);
  void update_barriers_after(const std::vector<WeftBarrier*> &after);
public:
  void update_happens_relationships(void);
protected:
  bool initialized;
  std::vector<WeftBarrier*> latest_before;
  std::vector<WeftBarrier*> earliest_after;
  std::vector<int> happens_before;
  std::vector<int> happens_after;
};

#endif // __RACE_H__
