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

#ifndef __INSTRUCTION_H__
#define __INSTRUCTION_H__

#include <map>
#include <string>
#include <vector>

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

// Special registers get special values
// Make these macros so they can be negative
#define WEFT_TID_X_REG    (-1)
#define WEFT_TID_Y_REG    (-2)
#define WEFT_TID_Z_REG    (-3)
#define WEFT_NTID_X_REG   (-4)
#define WEFT_NTID_Y_REG   (-5)
#define WEFT_NTID_Z_REG   (-6)
#define WEFT_LANE_REG     (-7)
#define WEFT_WARP_REG     (-8)
#define WEFT_NWARP_REG    (-9)
#define WEFT_CTA_X_REG    (-10)
#define WEFT_CTA_Y_REG    (-11)
#define WEFT_CTA_Z_REG    (-12)
#define WEFT_NCTA_X_REG   (-13)
#define WEFT_NCTA_Y_REG   (-14)
#define WEFT_NCTA_Z_REG   (-15)

#define SDDRINC (100000000)

enum PTXKind {
  PTX_SHARED_DECL,
  PTX_MOVE,
  PTX_RIGHT_SHIFT,
  PTX_LEFT_SHIFT,
  PTX_AND,
  PTX_OR,
  PTX_XOR,
  PTX_NOT,
  PTX_ADD,
  PTX_SUB,
  PTX_NEGATE,
  PTX_CONVERT,
  PTX_CONVERT_ADDRESS,
  PTX_BFE,
  PTX_MULTIPLY,
  PTX_MAD,
  PTX_SET_PREDICATE,
  PTX_SELECT_PREDICATE,
  PTX_BARRIER,
  PTX_SHARED_ACCESS,
  PTX_LABEL,
  PTX_BRANCH,
  PTX_UNIFORM_BRANCH,
  PTX_SHFL,
  PTX_EXIT,
  PTX_GLOBAL_DECL,
  PTX_GLOBAL_LOAD,
  PTX_LAST, // this one must be last
};

enum CompType {
  COMP_GT,
  COMP_GE,
  COMP_EQ,
  COMP_NE,
  COMP_LE,
  COMP_LT,
};

// Some helper methods
inline void split(std::vector<std::string> &results, 
                  const char *str, char c = ',') 
{
  do {
    const char *begin = str;
    while ((*str != ' ') && (*str != '\t') && 
           (*str != c) && (*str)) str++;

    std::string result(begin, str);
    if (!result.empty())
      results.push_back(result);
  } while (0 != *str++);
}

class Thread;
class PTXLabel;
class PTXBranch;
class PTXBarrier;
class WeftBarrier;
class WeftAccess;
class BarrierSync;
class BarrierArrive;
class SharedWrite;
class SharedRead;
class SharedStore;
class BarrierInstance;

class PTXInstruction {
public:
  PTXInstruction(void);
  PTXInstruction(PTXKind kind, int line_num);
  virtual~ PTXInstruction(void);
public:
  virtual PTXInstruction* emulate(Thread *thread) = 0;
  // Most instructions do the same thing, but some need
  // to override this behavior so make it virtual
  virtual PTXInstruction* emulate_warp(Thread **threads,
                                       ThreadState *thread_state,
                                       int &shared_access_id,
                                       SharedStore &store);
public:
  virtual bool is_label(void) const { return false; }
  virtual bool is_branch(void) const { return false; } 
  virtual bool is_barrier(void) const { return false; }
  virtual bool is_shuffle(void) const { return false; }
public:
  virtual PTXLabel* as_label(void) { return NULL; }
  virtual PTXBranch* as_branch(void) { return NULL; }
  virtual PTXBarrier* as_barrier(void) { return NULL; }
public:
  inline PTXKind get_kind(void) const { return kind; }
public:
  void set_next(PTXInstruction *next);
  void set_source_location(const char *file, int line);
public:
  static PTXInstruction* interpret(const std::string &line, int line_num);
  static const char* get_kind_name(PTXKind k);
public:
  static uint64_t compress_identifier(const char *buffer, size_t buffer_size);
  static void decompress_identifier(uint64_t id, char *buffer, size_t buffer_size);
public:
  const PTXKind kind;
  const int line_number;
protected:
  PTXInstruction *next;
public:
  const char *source_file;
  int source_line_number;
};

class PTXLabel: public PTXInstruction {
public:
  PTXLabel(const std::string &label, int line_num);
  PTXLabel(const PTXLabel &rhs) { assert(false); }
  virtual ~PTXLabel(void) { }
public:
  PTXLabel& operator=(const PTXLabel &rhs) { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread); 
  // Override for warp-synchronous execution
  virtual PTXInstruction* emulate_warp(Thread **threads,
                                       ThreadState *thread_state,
                                       int &shared_access_id,
                                       SharedStore &store);
public:
  virtual bool is_label(void) const { return true; }
public:
  virtual PTXLabel* as_label(void) { return this; }
public:
  void update_labels(std::map<std::string,PTXLabel*> &labels);
protected:
  std::string label;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXBranch : public PTXInstruction {
public:
  PTXBranch(const std::string &label, int line_num);
  PTXBranch(int64_t predicate, bool negate, const std::string &label, int line_num);
  PTXBranch(const PTXBranch &rhs) { assert(false); }
  virtual ~PTXBranch(void) { }
public:
  PTXBranch& operator=(const PTXBranch &rhs) { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
  // Override for warp-synchronous execution!
  virtual PTXInstruction* emulate_warp(Thread **threads,
                                       ThreadState *thread_state,
                                       int &shared_access_id,
                                       SharedStore &store);
public:
  virtual bool is_branch(void) const { return true; }
public:
  virtual PTXBranch* as_branch(void) { return this; }
public:
  void set_targets(const std::map<std::string,PTXLabel*> &labels);
protected:
  int64_t predicate;
  bool negate;
  std::string label;
  PTXLabel *target;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXSharedDecl : public PTXInstruction {
public:
  PTXSharedDecl(const std::string &name, int64_t address, int line_num);
  PTXSharedDecl(const PTXSharedDecl &rhs) { assert(false); }
  virtual ~PTXSharedDecl(void) { }
public:
  PTXSharedDecl& operator=(const PTXSharedDecl &rhs) 
    { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  std::string name;
  int64_t address;
public:
  static bool interpret(const std::string &line, int line_num, 
                        PTXInstruction *&result);
};

class PTXMove : public PTXInstruction {
public:
  PTXMove(int64_t dst, int64_t src, bool immediate, int line_num);
  PTXMove(int64_t dst, const std::string &src, int line_num);
  PTXMove(const PTXMove &rhs) { assert(false); }
  virtual ~PTXMove(void) { }
public:
  PTXMove& operator=(const PTXMove &rhs) { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  int64_t args[2];
  std::string source;
  bool immediate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXRightShift : public PTXInstruction {
public:
  PTXRightShift(int64_t zero, int64_t one, int64_t two, 
                bool immediate, int line_num);
  PTXRightShift(const PTXRightShift &rhs) { assert(false); }
  virtual ~PTXRightShift(void) { }
public:
  PTXRightShift& operator=(const PTXRightShift &rhs) 
    { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  int64_t args[3];
  bool immediate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXLeftShift : public PTXInstruction {
public:
  PTXLeftShift(int64_t zero, int64_t one, int64_t two, 
               bool immediate, int line_num);
  PTXLeftShift(const PTXLeftShift &rhs) { assert(false); }
  virtual ~PTXLeftShift(void) { }
public:
  PTXLeftShift& operator=(const PTXLeftShift &rhs) 
    { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  int64_t args[3];
  bool immediate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXAnd : public PTXInstruction {
public:
  PTXAnd(int64_t zero, int64_t one, int64_t two, 
         bool immediate, bool predicate, int line_num);
  PTXAnd(const PTXAnd &rhs) { assert(false); }
  virtual ~PTXAnd(void) { }
public:
  PTXAnd& operator=(const PTXAnd &rhs) { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  int64_t args[3];
  bool immediate;
  bool predicate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXOr : public PTXInstruction {
public:
  PTXOr(int64_t zero, int64_t one, int64_t two, 
        bool immediate, bool predicate, int line_num);
  PTXOr(const PTXOr &rhs) { assert(false); }
  virtual ~PTXOr(void) { }
public:
  PTXOr& operator=(const PTXOr &rhs) { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  int64_t args[3];
  bool immediate;
  bool predicate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXXor : public PTXInstruction {
public:
  PTXXor(int64_t zero, int64_t one, int64_t two, 
        bool immediate, bool predicate, int line_num);
  PTXXor(const PTXXor &rhs) { assert(false); }
  virtual ~PTXXor(void) { }
public:
  PTXXor& operator=(const PTXXor &rhs) { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  int64_t args[3];
  bool immediate;
  bool predicate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXNot : public PTXInstruction {
public:
  PTXNot(int64_t zero, int64_t one, bool predicate, int line_num);
  PTXNot(const PTXNot &rhs) { assert(false); }
  virtual ~PTXNot(void) { }
public:
  PTXNot& operator=(const PTXNot &rhs) { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  int64_t args[2];
  bool predicate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXAdd : public PTXInstruction {
public:
  PTXAdd(int64_t zero, int64_t one, int64_t two, 
         bool immediate, int line_num);
  PTXAdd(const PTXAdd &rhs) { assert(false); }
  virtual ~PTXAdd(void) { }
public:
  PTXAdd& operator=(const PTXAdd &rhs) { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  int64_t args[3];
  bool immediate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXSub : public PTXInstruction {
public:
  PTXSub(int64_t zero, int64_t one, int64_t two, 
         bool immediate, int line_num);
  PTXSub(const PTXSub &rhs) { assert(false); }
  virtual ~PTXSub(void) { }
public:
  PTXSub& operator=(const PTXSub &rhs) { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  int64_t args[3];
  bool immediate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXNeg : public PTXInstruction {
public:
  PTXNeg(int64_t zero, int64_t one, bool immediate, int line_num);
  PTXNeg(const PTXNeg &rhs) { assert(false); }
  virtual ~PTXNeg(void) { }
public:
  PTXNeg& operator=(const PTXNeg &rhs) { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  int64_t args[2];
  bool immediate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXMul : public PTXInstruction {
public:
  PTXMul(int64_t zero, int64_t one, int64_t two, 
         bool immediate, int line_num);
  PTXMul(const PTXMul &rhs) { assert(false); }
  virtual ~PTXMul(void) { }
public:
  PTXMul& operator=(const PTXMul &rhs) { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  int64_t args[3];
  bool immediate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXMad : public PTXInstruction {
public:
  PTXMad(int64_t args[4], bool immediates[4], int line_num);
  PTXMad(const PTXMad &rhs) { assert(false); }
  virtual ~PTXMad(void) { }
public:
  PTXMad& operator=(const PTXMad &rhs) { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  int64_t args[4];
  bool immediate[4];
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXSetPred : public PTXInstruction {
public:
  PTXSetPred(int64_t zero, int64_t one, int64_t two, bool immediate, 
             CompType comparison, int line_num);
  PTXSetPred(const PTXSetPred &rhs) { assert(false); }
  virtual ~PTXSetPred(void) { }
public:
  virtual PTXInstruction* emulate(Thread *thread);
public:
  PTXSetPred& operator=(const PTXSetPred &rhs) { assert(false); return *this; }
protected:
  int64_t args[3];
  CompType comparison;
  bool immediate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXSelectPred : public PTXInstruction {
public:
  PTXSelectPred(int64_t zero, int64_t one, int64_t two, int64_t three,
                bool negate, bool two_imm, bool three_imm, int line_num);
  PTXSelectPred(const PTXSelectPred &rhs) { assert(false); }
  virtual ~PTXSelectPred(void) { }
public:
  PTXSelectPred& operator=(const PTXSelectPred &rhs) { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  bool negate;
  int64_t predicate;
  int64_t args[3];
  bool immediate[2];
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXBarrier : public PTXInstruction {
public:
  PTXBarrier(int64_t name, int64_t count, bool sync, 
             bool name_imm, bool count_imm, int line_num);
  PTXBarrier(const PTXBarrier &rhs) { assert(false); }
  virtual ~PTXBarrier(void) { }
public:
  PTXBarrier& operator=(const PTXBarrier &rhs) { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
  // Override for warp-synchronous execution!
  virtual PTXInstruction* emulate_warp(Thread **threads,
                                       ThreadState *thread_state,
                                       int &shared_access_id,
                                       SharedStore &store);
public:
  virtual bool is_barrier(void) const { return true; }
  virtual PTXBarrier* as_barrier(void) { return this; }
  void update_count(unsigned arrival_count);
  int get_barrier_name(void) const { return name; }
protected:
  int64_t name, count;
  bool sync;
  bool name_immediate, count_immediate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXSharedAccess : public PTXInstruction {
public:
  PTXSharedAccess(int64_t addr, int64_t offset, bool write, 
                  bool has_arg, int64_t arg, bool immediate, int line_num);
  PTXSharedAccess(const std::string &name, int64_t offset, bool write,
                  bool has_arg, int64_t arg, bool immediate, int line_num);
  PTXSharedAccess(const PTXSharedAccess &rhs) { assert(false); }
  virtual ~PTXSharedAccess(void) { }
public:
  PTXSharedAccess& operator=(const PTXSharedAccess &rhs) 
    { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
  // Override for warp-synchronous execution!
  virtual PTXInstruction* emulate_warp(Thread **threads,
                                       ThreadState *thread_state,
                                       int &shared_access_id,
                                       SharedStore &store);
protected:
  bool has_name;
  std::string name;
  int64_t addr, offset, arg;
  bool write, has_arg, immediate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXConvert : public PTXInstruction {
public:
  PTXConvert(int64_t zero, int64_t one, int line_num);
  PTXConvert(const PTXConvert &rhs) { assert(false); }
  virtual ~PTXConvert(void) { }
public:
  PTXConvert& operator=(const PTXConvert &rhs) { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  int64_t src, dst;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXConvertAddress : public PTXInstruction {
public:
  PTXConvertAddress(int64_t zero, int64_t one, int line_num);
  PTXConvertAddress(int64_t zero, const std::string &name, int line_num);
  PTXConvertAddress(const PTXConvertAddress &rhs) { assert(false); }
  virtual ~PTXConvertAddress(void) { }
public:
  PTXConvertAddress& operator=(const PTXConvertAddress &rhs)
  { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  bool has_name;
  int64_t src, dst;
  std::string name;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXBitFieldExtract : public PTXInstruction {
public:
  PTXBitFieldExtract(int64_t args[4], bool immediates[4], int line_num);
  PTXBitFieldExtract(const PTXBitFieldExtract &rhs) { assert(false); }
  virtual ~PTXBitFieldExtract(void) { }
public:
  PTXBitFieldExtract& operator=(const PTXBitFieldExtract &rhs)
    { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  int64_t args[4];
  bool immediate[4];
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXShuffle : public PTXInstruction {
public:
  enum ShuffleKind {
    SHUFFLE_UP,
    SHUFFLE_DOWN,
    SHUFFLE_BUTTERFLY,
    SHUFFLE_IDX,
  };
public:
  PTXShuffle(ShuffleKind kind, int64_t args[4], bool immediates[4], int line_num);
  PTXShuffle(const PTXShuffle &rhs) { assert(false); }
  virtual ~PTXShuffle(void) { }
public:
  PTXShuffle& operator=(const PTXShuffle &rhs)
    { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
  // Override for warp-synchronous execution!
  virtual PTXInstruction* emulate_warp(Thread **threads,
                                       ThreadState *thread_state,
                                       int &shared_access_id,
                                       SharedStore &store);
  virtual bool is_shuffle(void) const { return true; }
protected:
  ShuffleKind kind;
  int64_t args[4];
  bool immediate[4];
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXExit : public PTXInstruction {
public:
  PTXExit(int line_num);
  PTXExit(int64_t predicate, bool negate, int line_num);
  PTXExit(const PTXExit &rhs) { assert(false); }
  virtual ~PTXExit(void) { }
public:
  PTXExit& operator=(const PTXExit &rhs)
    { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
  // Override for warp-synchronous execution!
  virtual PTXInstruction* emulate_warp(Thread **threads,
                                       ThreadState *thread_state,
                                       int &shared_access_id,
                                       SharedStore &store);
protected:
  bool has_predicate;
  bool negate;
  int64_t predicate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXGlobalDecl : public PTXInstruction {
public:
  PTXGlobalDecl(char *name, int *values, size_t size, int line_num);
  PTXGlobalDecl(const PTXGlobalDecl &rhs) { assert(false); }
  virtual ~PTXGlobalDecl(void) { free(name); free(values); }
public:
  PTXGlobalDecl& operator=(const PTXGlobalDecl &rhs) 
    { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  char *name;
  int *values;
  size_t size;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXGlobalLoad : public PTXInstruction {
public:
  PTXGlobalLoad(int64_t dst, int64_t addr, int line_num);
  PTXGlobalLoad(const PTXGlobalLoad &rhs) { assert(false); }
  virtual ~PTXGlobalLoad(void) { };
public:
  PTXGlobalLoad& operator=(const PTXGlobalLoad &rhs)
    { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  int64_t dst, addr;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class WeftInstruction {
public:
  WeftInstruction(PTXInstruction *instruction, Thread *thread);
  WeftInstruction(const WeftInstruction &rhs) : instruction(NULL), 
    thread(NULL), thread_line_number(-1) { assert(false); }
  virtual ~WeftInstruction(void) { }
public:
  WeftInstruction& operator=(const WeftInstruction &rhs)
    { assert(false); return *this; }
public:
  virtual bool is_barrier(void) const { return false; }
  virtual WeftBarrier* as_barrier(void) { return NULL; }
public:
  virtual bool is_access(void) const { return false; }
  virtual WeftAccess* as_access(void) { return NULL; }
public:
  virtual bool is_sync(void) const { return false; }
  virtual BarrierSync* as_sync(void) { return NULL; }
public:
  virtual bool is_arrive(void) const { return false; }
  virtual BarrierArrive* as_arrive(void) { return NULL; }
public:
  virtual bool is_write(void) const { return false; }
  virtual SharedWrite* as_write(void) { return NULL; }
public:
  virtual bool is_read(void) const { return false; }
  virtual SharedRead* as_read(void) { return NULL; }
public:
  void initialize_happens(Happens *happens);
  inline Happens* get_happens(void) const { return happens_relationship; }
public:
  virtual void print_instruction(FILE *target) = 0;
public:
  PTXInstruction *const instruction;
  Thread *const thread;
  const int thread_line_number;
protected:
  Happens *happens_relationship;
};

class WeftBarrier : public WeftInstruction {
public:
  WeftBarrier(int name, int count, PTXBarrier *bar, Thread *thread);
  WeftBarrier(const WeftBarrier &rhs) : WeftInstruction(NULL, NULL),
    name(0), count(0), barrier(NULL) { assert(false); }
  virtual ~WeftBarrier(void) { }
public:
  WeftBarrier& operator=(const WeftBarrier &rhs) { assert(false); return *this; }
public:
  virtual bool is_barrier(void) const { return true; }
  virtual WeftBarrier* as_barrier(void) { return this; }
public:
  void set_instance(BarrierInstance *instance);
  inline BarrierInstance* get_instance(void) const { return instance; }
public:
  virtual void print_instruction(FILE *target) = 0;
public:
  const int name;
  const int count;
  PTXBarrier *const barrier;
protected:
  BarrierInstance *instance;
};

class BarrierSync : public WeftBarrier {
public:
  BarrierSync(int name, int count, PTXBarrier *bar, Thread *thread);
  BarrierSync(const BarrierSync &rhs) : WeftBarrier(0, 0, NULL, NULL)
    { assert(false); }
  virtual ~BarrierSync(void) { }
public:
  BarrierSync& operator=(const BarrierSync &rhs) { assert(false); return *this; }
public:
  virtual bool is_sync(void) const { return true; }
  virtual BarrierSync* as_sync(void) { return this; }
  virtual void print_instruction(FILE *target);
};

class BarrierArrive : public WeftBarrier {
public:
  BarrierArrive(int name, int count, PTXBarrier *bar, Thread *thread);
  BarrierArrive(const BarrierArrive &rhs) : WeftBarrier(0, 0, NULL, NULL)
    { assert(false); }
  virtual ~BarrierArrive(void) { }
public:
  BarrierArrive& operator=(const BarrierArrive &rhs) { assert(false); return *this; }
public:
  virtual bool is_arrive(void) const { return true; }
  virtual BarrierArrive* as_arrive(void) { return this; }
  virtual void print_instruction(FILE *target);
};

class WeftAccess : public WeftInstruction {
public:
  WeftAccess(int address, PTXSharedAccess *access, Thread *thread, int access_id);
  WeftAccess(const WeftAccess &rhs) : WeftInstruction(NULL, NULL),
    address(0), access(NULL), access_id(-1) { assert(false); }
  virtual ~WeftAccess(void) { }
public:
  WeftAccess& operator=(const WeftAccess &rhs) { assert(false); return *this; }
public:
  virtual bool is_access(void) const { return true; }
  virtual WeftAccess* as_access(void) { return this; }
public:
  bool has_happens_relationship(WeftAccess *other);
  bool is_warp_synchronous(WeftAccess *other);
public:
  virtual void print_instruction(FILE *target) = 0;
public:
  const int address;
  PTXSharedAccess *const access;
  const int access_id; // for warp-synchronous execution
};

class SharedWrite : public WeftAccess {
public:
  SharedWrite(int address, PTXSharedAccess *access, 
              Thread *thread, int access_id = -1);
  SharedWrite(const SharedWrite &rhs) : WeftAccess(0, NULL, NULL, -1)
    { assert(false); }
  virtual ~SharedWrite(void) { }
public:
  SharedWrite& operator=(const SharedWrite &rhs) { assert(false); return *this; }
public:
  virtual bool is_write(void) const { return true; }
  virtual SharedWrite* as_write(void) { return this; }
  virtual void print_instruction(FILE *target);
};

class SharedRead : public WeftAccess {
public:
  SharedRead(int address, PTXSharedAccess *access, 
             Thread *thread, int access_id = -1);
  SharedRead(const SharedRead &rhs) : WeftAccess(0, NULL, NULL, -1)
    { assert(false); }
  virtual ~SharedRead(void) { }
public:
  SharedRead& operator=(const SharedRead &rhs) { assert(false); return *this; }
public:
  virtual bool is_read(void) const { return true; }
  virtual SharedRead* as_read(void) { return this; }
  virtual void print_instruction(FILE *target);
};

#endif // __INSTRUCTION_H__
