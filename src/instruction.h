
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
  PTXLabel(const std::string &line, int line_num);
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
};

class PTXBranch : public PTXInstruction {
public:
  PTXBranch(const std::string &line, int line_num);
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
  PTXMove(const std::string &dst, const std::string &src, int line_num);
  PTXMove(const std::string &dst, int immediate, int line_num);
  PTXMove(const PTXMove &rhs) { assert(false); }
  virtual ~PTXMove(void) { }
public:
  PTXMove& operator=(const PTXMove &rhs) { assert(false); return *this; }
protected:
  std::string dst;
  std::string src;
  bool immediate;
  int immediate_value;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXRightShift : public PTXInstruction {
public:
  PTXRightShift(const std::string &dst, const std::string &src,
                int shift_value, int line_num);
  PTXRightShift(const PTXRightShift &rhs) { assert(false); }
  virtual ~PTXRightShift(void) { }
public:
  PTXRightShift& operator=(const PTXRightShift &rhs) 
    { assert(false); return *this; }
protected:
  std::string dst;
  std::string src; 
  int shift_value;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

class PTXLeftShift : public PTXInstruction {
public:
  PTXLeftShift(const std::string &dst, const std::string &src,
               int shift_value, int line_num);
  PTXLeftShift(const PTXRightShift &rhs) { assert(false); }
  virtual ~PTXLeftShift(void) { }
public:
  PTXLeftShift& operator=(const PTXLeftShift &rhs) 
    { assert(false); return *this; }
protected:
  std::string dst;
  std::string src; 
  int shift_value;
public:
  static bool interpret(const std::string &line, int line_num,
                        PTXInstruction *&result);
};

#endif // __INSTRUCTION_H__
