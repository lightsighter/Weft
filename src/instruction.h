
#ifndef __INSTRUCTION_H__
#define __INSTRUCTION_H__

#include <map>
#include <string>
#include <vector>

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

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

class PTXLabel;
class PTXBranch;

class PTXInstruction {
public:
  PTXInstruction(void);
  PTXInstruction(PTXKind kind, int line_num);
  virtual~ PTXInstruction(void);
public:
  virtual bool is_label(void) const { return false; }
  virtual bool is_branch(void) const { return false; } 
public:
  virtual PTXLabel* as_label(void) { return NULL; }
  virtual PTXBranch* as_branch(void) { return NULL; }
public:
  inline PTXKind get_kind(void) const { return kind; }
public:
  void set_next(PTXInstruction *next);
public:
  static PTXInstruction* interpret(const std::string &line, int line_num);
  static const char* get_kind_name(PTXKind k);
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
  PTXBranch(int predicate, const std::string &label, int line_num);
  PTXBranch(const PTXBranch &rhs) { assert(false); }
  virtual ~PTXBranch(void) { }
public:
  PTXBranch& operator=(const PTXBranch &rhs) { assert(false); return *this; }
public:
  virtual bool is_branch(void) const { return true; }
public:
  virtual PTXBranch* as_branch(void) { return this; }
public:
  void set_targets(const std::map<std::string,PTXInstruction*> &labels);
protected:
  int predicate;
  std::string label;
  PTXInstruction *target;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXSharedDecl : public PTXInstruction {
public:
  PTXSharedDecl(const std::string &name, int address, int line_num);
  PTXSharedDecl(const PTXSharedDecl &rhs) { assert(false); }
  virtual ~PTXSharedDecl(void) { }
public:
  PTXSharedDecl& operator=(const PTXSharedDecl &rhs) 
    { assert(false); return *this; }
protected:
  std::string name;
  int address;
public:
  static bool interpret(const std::string &line, int line_num, 
                        PTXInstruction *&result);
};

class PTXMove : public PTXInstruction {
public:
  PTXMove(int dst, int src, bool immediate, int line_num);
  PTXMove(int dst, const std::string &src, int line_num);
  PTXMove(const PTXMove &rhs) { assert(false); }
  virtual ~PTXMove(void) { }
public:
  PTXMove& operator=(const PTXMove &rhs) { assert(false); return *this; }
protected:
  int args[2];
  std::string source;
  bool immediate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXRightShift : public PTXInstruction {
public:
  PTXRightShift(int zero, int one, int two, bool immediate, int line_num);
  PTXRightShift(const PTXRightShift &rhs) { assert(false); }
  virtual ~PTXRightShift(void) { }
public:
  PTXRightShift& operator=(const PTXRightShift &rhs) 
    { assert(false); return *this; }
protected:
  int args[3];
  bool immediate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXLeftShift : public PTXInstruction {
public:
  PTXLeftShift(int zero, int one, int two, bool immediate, int line_num);
  PTXLeftShift(const PTXLeftShift &rhs) { assert(false); }
  virtual ~PTXLeftShift(void) { }
public:
  PTXLeftShift& operator=(const PTXLeftShift &rhs) 
    { assert(false); return *this; }
protected:
  int args[3];
  bool immediate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXAnd : public PTXInstruction {
public:
  PTXAnd(int zero, int one, int two, bool immediate, int line_num);
  PTXAnd(const PTXAnd &rhs) { assert(false); }
  virtual ~PTXAnd(void) { }
public:
  PTXAnd& operator=(const PTXAnd &rhs) { assert(false); return *this; }
protected:
  int args[3];
  bool immediate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXOr : public PTXInstruction {
public:
  PTXOr(int zero, int one, int two, bool immediate, int line_num);
  PTXOr(const PTXOr &rhs) { assert(false); }
  virtual ~PTXOr(void) { }
public:
  PTXOr& operator=(const PTXOr &rhs) { assert(false); return *this; }
protected:
  int args[3];
  bool immediate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXAdd : public PTXInstruction {
public:
  PTXAdd(int zero, int one, int two, bool immediate, int line_num);
  PTXAdd(const PTXAdd &rhs) { assert(false); }
  virtual ~PTXAdd(void) { }
public:
  PTXAdd& operator=(const PTXAdd &rhs) { assert(false); return *this; }
protected:
  int args[3];
  bool immediate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXSub : public PTXInstruction {
public:
  PTXSub(int zero, int one, int two, bool immediate, int line_num);
  PTXSub(const PTXSub &rhs) { assert(false); }
  virtual ~PTXSub(void) { }
public:
  PTXSub& operator=(const PTXSub &rhs) { assert(false); return *this; }
protected:
  int args[3];
  bool immediate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXNeg : public PTXInstruction {
public:
  PTXNeg(int zero, int one, bool immediate, int line_num);
  PTXNeg(const PTXNeg &rhs) { assert(false); }
  virtual ~PTXNeg(void) { }
public:
  PTXNeg& operator=(const PTXNeg &rhs) { assert(false); return *this; }
protected:
  int args[2];
  bool immediate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXMul : public PTXInstruction {
public:
  PTXMul(int zero, int one, int two, bool immediate, int line_num);
  PTXMul(const PTXMul &rhs) { assert(false); }
  virtual ~PTXMul(void) { }
public:
  PTXMul& operator=(const PTXMul &rhs) { assert(false); return *this; }
protected:
  int args[3];
  bool immediate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXMad : public PTXInstruction {
public:
  PTXMad(int args[4], bool immediates[4], int line_num);
  PTXMad(const PTXMad &rhs) { assert(false); }
  virtual ~PTXMad(void) { }
public:
  PTXMad& operator=(const PTXMad &rhs) { assert(false); return *this; }
protected:
  int args[4];
  bool immediate[4];
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXSetPred : public PTXInstruction {
public:
  PTXSetPred(int zero, int one, int two, bool immediate, 
             CompType comparison, int line_num);
  PTXSetPred(const PTXSetPred &rhs) { assert(false); }
  virtual ~PTXSetPred(void) { }
public:
  PTXSetPred& operator=(const PTXSetPred &rhs) { assert(false); return *this; }
protected:
  int args[3];
  CompType comparison;
  bool immediate;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXSelectPred : public PTXInstruction {
public:
  PTXSelectPred(int zero, int one, int two, int three,
                bool two_imm, bool three_imm, int line_num);
  PTXSelectPred(const PTXSelectPred &rhs) { assert(false); }
  virtual ~PTXSelectPred(void) { }
public:
  PTXSelectPred& operator=(const PTXSelectPred &rhs) { assert(false); return *this; }
protected:
  int predicate;
  int args[3];
  bool immediate[2];
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXBarrier : public PTXInstruction {
public:
  PTXBarrier(int name, int count, bool sync, int line_num);
  PTXBarrier(const PTXBarrier &rhs) { assert(false); }
  virtual ~PTXBarrier(void) { }
public:
  PTXBarrier& operator=(const PTXBarrier &rhs) { assert(false); return *this; }
protected:
  int name, count;
  bool sync;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXSharedAccess : public PTXInstruction {
public:
  PTXSharedAccess(int addr, int offset, bool write, int line_num);
  PTXSharedAccess(const PTXSharedAccess &rhs) { assert(false); }
  virtual ~PTXSharedAccess(void) { }
public:
  PTXSharedAccess& operator=(const PTXSharedAccess &rhs) 
    { assert(false); return *this; }
protected:
  int addr, offset;
  bool write;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXConvert : public PTXInstruction {
public:
  PTXConvert(int zero, int one, int line_num);
  PTXConvert(const PTXConvert &rhs) { assert(false); }
  virtual ~PTXConvert(void) { }
public:
  PTXConvert& operator=(const PTXConvert &rhs) { assert(false); return *this; }
protected:
  int src, dst;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

#endif // __INSTRUCTION_H__
