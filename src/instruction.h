
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
#define WEFT_TID_REG    (-1)
#define WEFT_CTA_REG    (-2)

enum PTXKind {
  PTX_SHARED_DECL,
  PTX_MOVE,
  PTX_RIGHT_SHIFT,
  PTX_LEFT_SHIFT,
  PTX_AND,
  PTX_OR,
  PTX_ADD,
  PTX_SUB,
  PTX_NEGATE,
  PTX_CONVERT,
  PTX_CONVERT_ADDRESS,
  PTX_MULTIPLY,
  PTX_MAD,
  PTX_SET_PREDICATE,
  PTX_SELECT_PREDICATE,
  PTX_BARRIER,
  PTX_SHARED_ACCESS,
  PTX_LABEL,
  PTX_BRANCH,
  PTX_UNIFORM_BRANCH,
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

class Thread;
class PTXLabel;
class PTXBranch;
class PTXBarrier;

class PTXInstruction {
public:
  PTXInstruction(void);
  PTXInstruction(PTXKind kind, int line_num);
  virtual~ PTXInstruction(void);
public:
  virtual PTXInstruction* emulate(Thread *thread) = 0;
public:
  virtual bool is_label(void) const { return false; }
  virtual bool is_branch(void) const { return false; } 
  virtual bool is_barrier(void) const { return false; }
public:
  virtual PTXLabel* as_label(void) { return NULL; }
  virtual PTXBranch* as_branch(void) { return NULL; }
  virtual PTXBarrier* as_barrier(void) { return NULL; }
public:
  inline PTXKind get_kind(void) const { return kind; }
public:
  void set_next(PTXInstruction *next);
public:
  static PTXInstruction* interpret(const std::string &line, int line_num);
  static const char* get_kind_name(PTXKind k);
public:
  static uint64_t compress_identifier(const char *buffer, size_t buffer_size);
  static void decompress_identifier(uint64_t id, char *buffer, size_t buffer_size);
protected:
  const PTXKind kind;
  const int line_number;
  PTXInstruction *next;
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
public:
  virtual bool is_label(void) const { return true; }
public:
  virtual PTXLabel* as_label(void) { return this; }
public:
  void update_labels(std::map<std::string,PTXInstruction*> &labels);
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
public:
  virtual bool is_branch(void) const { return true; }
public:
  virtual PTXBranch* as_branch(void) { return this; }
public:
  void set_targets(const std::map<std::string,PTXInstruction*> &labels);
protected:
  int64_t predicate;
  bool negate;
  std::string label;
  PTXInstruction *target;
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
         bool immediate, int line_num);
  PTXAnd(const PTXAnd &rhs) { assert(false); }
  virtual ~PTXAnd(void) { }
public:
  PTXAnd& operator=(const PTXAnd &rhs) { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  int64_t args[3];
  bool immediate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXOr : public PTXInstruction {
public:
  PTXOr(int64_t zero, int64_t one, int64_t two, 
        bool immediate, int line_num);
  PTXOr(const PTXOr &rhs) { assert(false); }
  virtual ~PTXOr(void) { }
public:
  PTXOr& operator=(const PTXOr &rhs) { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  int64_t args[3];
  bool immediate;
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
  PTXBarrier(int64_t name, int64_t count, bool sync, int line_num);
  PTXBarrier(const PTXBarrier &rhs) { assert(false); }
  virtual ~PTXBarrier(void) { }
public:
  PTXBarrier& operator=(const PTXBarrier &rhs) { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
public:
  virtual bool is_barrier(void) const { return true; }
  virtual PTXBarrier* as_barrier(void) { return this; }
  void update_count(unsigned arrival_count);
protected:
  int64_t name, count;
  bool sync;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXSharedAccess : public PTXInstruction {
public:
  PTXSharedAccess(int64_t addr, int64_t offset, bool write, int line_num);
  PTXSharedAccess(const PTXSharedAccess &rhs) { assert(false); }
  virtual ~PTXSharedAccess(void) { }
public:
  PTXSharedAccess& operator=(const PTXSharedAccess &rhs) 
    { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  int64_t addr, offset;
  bool write;
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
  PTXConvertAddress(const PTXConvertAddress &rhs) { assert(false); }
  virtual ~PTXConvertAddress(void) { }
public:
  PTXConvertAddress& operator=(const PTXConvertAddress &rhs)
  { assert(false); return *this; }
public:
  virtual PTXInstruction* emulate(Thread *thread);
protected:
  int64_t src, dst;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class WeftInstruction {
public:
  WeftInstruction(PTXInstruction *instruction, Thread *thread);
  WeftInstruction(const WeftInstruction &rhs) 
    : instruction(NULL), thread(NULL) { assert(false); }
  virtual ~WeftInstruction(void) { }
public:
  WeftInstruction& operator=(const WeftInstruction &rhs)
    { assert(false); return *this; }
public:
  PTXInstruction *const instruction;
  Thread *const thread;
};

class BarrierWait : public WeftInstruction {
public:
  BarrierWait(int name, int count, PTXBarrier *bar, Thread *thread);
  BarrierWait(const BarrierWait &rhs) : WeftInstruction(NULL, NULL), 
    name(0), count(0), barrier(NULL) { assert(false); }
  virtual ~BarrierWait(void) { }
public:
  BarrierWait& operator=(const BarrierWait &rhs) { assert(false); return *this; }
public:
  const int name;
  const int count;
  PTXBarrier *const barrier;
};

class BarrierArrive : public WeftInstruction {
public:
  BarrierArrive(int name, int count, PTXBarrier *bar, Thread *thread);
  BarrierArrive(const BarrierArrive &rhs) : WeftInstruction(NULL, NULL), 
    name(0), count(0), barrier(NULL) { assert(false); }
  virtual ~BarrierArrive(void) { }
public:
  BarrierArrive& operator=(const BarrierArrive &rhs) { assert(false); return *this; }
public:
  const int name;
  const int count;
  PTXBarrier *const barrier;
};

class SharedWrite : public WeftInstruction {
public:
  SharedWrite(int address, PTXSharedAccess *access, Thread *thread);
  SharedWrite(const SharedWrite &rhs) : WeftInstruction(NULL, NULL),
    address(0), access(NULL) { assert(false); }
  virtual ~SharedWrite(void) { }
public:
  SharedWrite& operator=(const SharedWrite &rhs) { assert(false); return *this; }
public:
  const int address;
  PTXSharedAccess *const access;
};

class SharedRead : public WeftInstruction {
public:
  SharedRead(int address, PTXSharedAccess *access, Thread *thread);
  SharedRead(const SharedRead &rhs) : WeftInstruction(NULL, NULL),
    address(0), access(NULL) { assert(false); }
  virtual ~SharedRead(void) { }
public:
  SharedRead& operator=(const SharedRead &rhs) { assert(false); return *this; }
public:
  const int address;
  PTXSharedAccess *const access;
};

#endif // __INSTRUCTION_H__
