
#include "instruction.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

#define SDDRINC (100000000)

inline size_t count(const std::string &base, const std::string &to_find)
{
  size_t result = 0;
  std::string::size_type start = 0;
  while ((start = base.find(to_find, start)) !=
          std::string::npos)
  {
    result++;
    start += to_find.length();
  }
  return result;
}

inline bool contains(const std::string &s1, const std::string &s2)
{
  return (s1.find(s2) != std::string::npos);
}

inline void split(std::vector<std::string> &results, 
                  const char *str, char c = ' ') 
{
  do {
    const char *begin = str;
    while ((*str != c) && (*str != '\t') && (*str)) str++;

    std::string result(begin, str);
    if (!result.empty())
      results.push_back(result);
  } while (0 != *str++);
}

inline int convert_predicate(const std::string &pred)
{
  int result = 0;
  const char *ptr = pred.c_str();
  for (int i = 0; i < pred.size(); i++)
  {
    int temp = (int)ptr[i];
    assert(((48 <= temp) && (temp <= 57)) ||
           ((97 <= temp) && (temp <= 122)));
    if (temp >= 97)
      temp -= 87;
    else
      temp -= 48;
    assert((0 <= temp) && (temp < 36));
    int scale = 1;
    for (int j = 0; j < i; j++)
      scale *= 36;
    result += (temp * scale);
  }
  return result;
}

PTXInstruction::PTXInstruction(void)
  : kind(PTX_LAST), line_number(0), next(NULL)
{
  // should never be called
  assert(false);
}

PTXInstruction::PTXInstruction(PTXKind k, int line_num)
  : kind(k), line_number(line_num), next(NULL)
{
}

PTXInstruction::~PTXInstruction(void)
{
}

void PTXInstruction::set_next(PTXInstruction *n)
{
  assert(n != NULL);
  assert(next == NULL);
  next = n;
}

/*static*/ 
PTXInstruction* PTXInstruction::interpret(const std::string &line, int line_num)
{
  PTXInstruction *result = NULL;
  if (PTXBranch::interpret(line, line_num, result))
    return result;
  if (PTXLabel::interpret(line, line_num, result))
    return result;
  if (PTXSharedDecl::interpret(line, line_num, result))
    return result; 
  if (PTXMove::interpret(line, line_num, result))
    return result;
  if (PTXRightShift::interpret(line, line_num, result))
    return result;
  if (PTXLeftShift::interpret(line, line_num, result))
    return result;
  if (PTXAnd::interpret(line, line_num, result))
    return result;
  if (PTXOr::interpret(line, line_num, result))
    return result;
  if (PTXAdd::interpret(line, line_num, result))
    return result;
  if (PTXSub::interpret(line, line_num, result))
    return result;
  if (PTXNeg::interpret(line, line_num, result))
    return result;
  if (PTXMul::interpret(line, line_num, result))
    return result;
  if (PTXMad::interpret(line, line_num, result))
    return result;
  if (PTXSetPred::interpret(line, line_num, result))
    return result;
  if (PTXSelectPred::interpret(line, line_num, result))
    return result;
  if (PTXBarrier::interpret(line, line_num, result))
    return result;
  if (PTXSharedAccess::interpret(line, line_num, result))
    return result;
  if (PTXConvert::interpret(line, line_num, result))
    return result;
  return result;
}

/*static*/
const char* PTXInstruction::get_kind_name(PTXKind kind)
{
  switch (kind)
  {
    case PTX_SHARED_DECL:
      return "Shared Memory Declaration";
    case PTX_MOVE:
      return "Move";
    case PTX_RIGHT_SHIFT:
      return "Right Shift";
    case PTX_LEFT_SHIFT:
      return "Left Shift";
    case PTX_AND:
      return "Logical And";
    case PTX_OR:
      return "Logical Or";
    case PTX_ADD:
      return "Integer Add";
    case PTX_SUB:
      return "Integer Subtract";
    case PTX_NEGATE:
      return "Negation";
    case PTX_CONVERT:
      return "Convert";
    case PTX_MULTIPLY:
      return "Integer Multiply";
    case PTX_MAD:
      return "Integer Fused Multiply Add";
    case PTX_SET_PREDICATE:
      return "Set Predicate";
    case PTX_SELECT_PREDICATE:
      return "Select Predicate";
    case PTX_BARRIER:
      return "Barrier";
    case PTX_SHARED_ACCESS:
      return "Shared Memory Access";
    case PTX_LABEL:
      return "Label";
    case PTX_BRANCH:
      return "Branch";
    case PTX_UNIFORM_BRANCH:
      return "Uniform Branch";
    default:
      assert(false);
  }
  return NULL;
}

PTXLabel::PTXLabel(const std::string &l, int line_num)
  : PTXInstruction(PTX_LABEL, line_num), label(l)
{
}

void PTXLabel::update_labels(std::map<std::string,PTXInstruction*> &labels)
{
  std::map<std::string,PTXInstruction*>::const_iterator finder = 
    labels.find(label);
  assert(finder == labels.end());
  labels[label] = this;
}

/*static*/
bool PTXLabel::interpret(const std::string &line, int line_num,
                         PTXInstruction *&result)
{
  if (line.find(":") != std::string::npos)
  {
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    assert(tokens.size() == 1);
    std::string label = tokens[0].substr(0, tokens[0].size() - 1);
    result = new PTXLabel(label, line_num);
    return true;
  }
  return false;
}

PTXBranch::PTXBranch(const std::string &l, int line_num)
  : PTXInstruction(PTX_BRANCH, line_num), predicate(0), label(l), target(NULL)
{
}

PTXBranch::PTXBranch(int p, bool n, const std::string &l, int line_num)
  : PTXInstruction(PTX_BRANCH, line_num), 
    predicate(p), negate(n), label(l), target(NULL)
{
}

void PTXBranch::set_targets(const std::map<std::string,PTXInstruction*> &labels)
{
  std::map<std::string,PTXInstruction*>::const_iterator finder =
    labels.find(label);
  assert(finder != labels.end());
  assert(target == NULL);
  target = finder->second;
}

/*static*/
bool PTXBranch::interpret(const std::string &line, int line_num,
                          PTXInstruction *&result)
{
  if (line.find("bra") != std::string::npos)
  {
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    if (tokens.size() == 3)
    {
      bool negate = (tokens[0].find("!") != std::string::npos);
      std::string arg1 = tokens[0].substr((negate ? 3 : 2), 
                         tokens[0].size() - (negate ? 4 : 3));
      int predicate = convert_predicate(arg1);
      std::string arg2 = tokens[2].substr(0, tokens[2].size() - 1);
      result = new PTXBranch(predicate, negate, arg2, line_num);
    }
    else if (tokens.size() == 2)
    {
      assert(line.find("bra.uni") != std::string::npos);
      std::string arg = tokens[1].substr(0, tokens[1].size() - 1);
      result = new PTXBranch(arg, line_num);
    }
    else
      assert(false);
    return true;
  }
  return false;
}

PTXSharedDecl::PTXSharedDecl(const std::string &n, int addr, int line_num)
  : PTXInstruction(PTX_SHARED_DECL, line_num), name(n), address(addr)
{
}

/*static*/
bool PTXSharedDecl::interpret(const std::string &line, int line_num,
                              PTXInstruction *&result)
{
  static int shared_offset = 1;
  if ((line.find(".shared") != std::string::npos) &&
      (line.find(".align") != std::string::npos))
  {
    int start = line.find("_");
    std::string name = line.substr(start, line.find("[") - start);
    // This is just an approximation to stride all 
    // the shared memory allocations far away from each other
    int address = shared_offset * SDDRINC;
    shared_offset++;
    result = new PTXSharedDecl(name, address, line_num);
    return true;
  }
  return false;
}

PTXMove::PTXMove(int dst, int src, bool imm, int line_num)
  : PTXInstruction(PTX_MOVE, line_num), immediate(imm)
{
  args[0] = dst;
  args[1] = src;
}

PTXMove::PTXMove(int dst, const std::string &src, int line_num)
  : PTXInstruction(PTX_MOVE, line_num), immediate(false)
{
  args[0] = dst;
  source = src;
}

/*static*/
bool PTXMove::interpret(const std::string &line, int line_num,
                        PTXInstruction *&result)
{
  if (line.find("mov.") != std::string::npos)
  {
    if (count(line, "%") == 2)
    {
      int start_arg1 = line.find("%");
      int end_arg1 = line.find(",");
      int start_arg2 = line.find("%", end_arg1);
      int arg1 = strtol(line.substr(start_arg1+1).c_str(), NULL, 10);
      int arg2 = strtol(line.substr(start_arg2+1).c_str(), NULL, 10);
      result = new PTXMove(arg1, arg2, false/*immediate*/, line_num);
      return true;
    }
    else if (contains(line, "cuda"))
    {
      int start_arg1 = line.find("%");
      int end_arg1 = line.find(",");
      int start_arg2 = line.find("_", end_arg1);
      int end_arg2 = line.find(";");
      int arg1 = strtol(line.substr(start_arg1+1).c_str(), NULL, 10);
      std::string arg2 = line.substr(start_arg2, end_arg2 - start_arg2);
      result = new PTXMove(arg1, arg2, line_num);
      return true;
    }
    else if (count(line, "%") == 1)
    {
      std::vector<std::string> tokens;
      split(tokens, line.c_str()); 
      assert(tokens.size() == 3);
      int arg1 = strtol(tokens[1].substr(1).c_str(), NULL, 10);
      int arg2 = strtol(tokens[2].c_str(), NULL, 10);
      result = new PTXMove(arg1, arg2, true/*immediate*/, line_num);
      return true;
    }
  }
  return false;
}

PTXRightShift::PTXRightShift(int one, int two, int three, 
                             bool imm, int line_num)
 : PTXInstruction(PTX_RIGHT_SHIFT, line_num), immediate(imm)
{
  args[0] = one;
  args[1] = two;
  args[2] = three;
}

/*static*/
bool PTXRightShift::interpret(const std::string &line, int line_num,
                              PTXInstruction *&result)
{
  if (line.find("shr.") != std::string::npos)
  {
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    assert(tokens.size() == 4);
    int arg1 = strtol(tokens[1].c_str()+1, NULL, 10);
    int arg2 = strtol(tokens[2].c_str()+1, NULL, 10);
    const size_t regs = count(line, "%");
    assert((regs == 2) || (regs == 3));
    const bool immediate = (regs == 2);
    int arg3 = strtol(tokens[3].c_str() + (immediate ? 0 : 1), NULL, 10);
    result = new PTXRightShift(arg1, arg2, arg3, immediate, line_num);
    return true;
  }
  return false;
}

PTXLeftShift::PTXLeftShift(int one, int two, int three, 
                             bool imm, int line_num)
 : PTXInstruction(PTX_LEFT_SHIFT, line_num), immediate(imm)
{
  args[0] = one;
  args[1] = two;
  args[2] = three;
}

/*static*/
bool PTXLeftShift::interpret(const std::string &line, int line_num,
                              PTXInstruction *&result)
{
  if (line.find("shl.") != std::string::npos)
  {
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    assert(tokens.size() == 4);
    int arg1 = strtol(tokens[1].c_str()+1, NULL, 10);
    int arg2 = strtol(tokens[2].c_str()+1, NULL, 10);
    const size_t regs = count(line, "%");
    assert((regs == 2) || (regs == 3));
    const bool immediate = (regs == 2);
    int arg3 = strtol(tokens[3].c_str() + (immediate ? 0 : 1), NULL, 10);
    result = new PTXLeftShift(arg1, arg2, arg3, immediate, line_num);
    return true;
  }
  return false;
}

PTXAnd::PTXAnd(int zero, int one, int two, bool imm, int line_num)
  : PTXInstruction(PTX_AND, line_num), immediate(imm)
{
  args[0] = zero;
  args[1] = one;
  args[2] = two;
}

/*static*/
bool PTXAnd::interpret(const std::string &line, int line_num,
                       PTXInstruction *&result)
{
  if (line.find("and.b") != std::string::npos)
  {
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    assert(tokens.size() == 4);
    int arg1 = strtol(tokens[1].c_str()+1, NULL, 10);
    int arg2 = strtol(tokens[2].c_str()+1, NULL, 10);
    const size_t regs = count(line, "%");
    assert((regs == 2) || (regs == 3));
    const bool immediate = (regs == 2);
    int arg3 = strtol(tokens[3].c_str() + (immediate ? 0 : 1), NULL, 10);
    result = new PTXAnd(arg1, arg2, arg3, immediate, line_num);
    return true;
  }
  return false;
}

PTXOr::PTXOr(int zero, int one, int two, bool imm, int line_num)
  : PTXInstruction(PTX_OR, line_num), immediate(imm)
{
  args[0] = zero;
  args[1] = one;
  args[2] = two;
}

/*static*/
bool PTXOr::interpret(const std::string &line, int line_num,
                      PTXInstruction *&result)
{
  if (line.find("or.b") != std::string::npos)
  {
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    assert(tokens.size() == 4);
    int arg1 = strtol(tokens[1].c_str()+1, NULL, 10);
    int arg2 = strtol(tokens[2].c_str()+1, NULL, 10);
    const size_t regs = count(line, "%");
    assert((regs == 2) || (regs == 3));
    const bool immediate = (regs == 2);
    int arg3 = strtol(tokens[3].c_str() + (immediate ? 0 : 1), NULL, 10);
    result = new PTXOr(arg1, arg2, arg3, immediate, line_num);
    return true;
  }
  return false;
}

PTXAdd::PTXAdd(int zero, int one, int two, bool imm, int line_num)
  : PTXInstruction(PTX_ADD, line_num), immediate(imm)
{
  args[0] = zero;
  args[1] = one;
  args[2] = two;
}

/*static*/
bool PTXAdd::interpret(const std::string &line, int line_num,
                       PTXInstruction *&result)
{
  if (line.find("add.") != std::string::npos)
  {
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    assert(tokens.size() == 4);
    int arg1 = strtol(tokens[1].c_str()+1, NULL, 10);
    int arg2 = strtol(tokens[2].c_str()+1, NULL, 10);
    const size_t regs = count(line, "%");
    assert((regs == 2) || (regs == 3));
    const bool immediate = (regs == 2);
    int arg3 = strtol(tokens[3].c_str() + (immediate ? 0 : 1), NULL, 10);
    result = new PTXAdd(arg1, arg2, arg3, immediate, line_num);
    return true;
  }
  return false;
}

PTXSub::PTXSub(int zero, int one, int two, bool imm, int line_num)
  : PTXInstruction(PTX_SUB, line_num), immediate(imm)
{
  args[0] = zero;
  args[1] = one;
  args[2] = two;
}

/*static*/
bool PTXSub::interpret(const std::string &line, int line_num,
                       PTXInstruction *&result)
{
  if (line.find("sub.") != std::string::npos)
  {
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    assert(tokens.size() == 4);
    int arg1 = strtol(tokens[1].c_str()+1, NULL, 10);
    int arg2 = strtol(tokens[2].c_str()+1, NULL, 10);
    const size_t regs = count(line, "%");
    assert((regs == 2) || (regs == 3));
    const bool immediate = (regs == 2);
    int arg3 = strtol(tokens[3].c_str() + (immediate ? 0 : 1), NULL, 10);
    result = new PTXSub(arg1, arg2, arg3, immediate, line_num);
    return true;
  }
  return false;
}

PTXNeg::PTXNeg(int zero, int one, bool imm, int line_num)
  : PTXInstruction(PTX_NEGATE, line_num), immediate(imm)
{
  args[0] = zero;
  args[1] = one;
}

/*static*/
bool PTXNeg::interpret(const std::string &line, int line_num,
                       PTXInstruction *&result)
{
  if (line.find("neg.") != std::string::npos)
  {
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    assert(tokens.size() == 3);
    int arg1 = strtol(tokens[1].c_str()+1, NULL, 10);
    const size_t regs = count(line, "%");
    assert((regs == 1) || (regs == 2));
    const bool immediate = (regs == 1);
    int arg2 = strtol(tokens[2].c_str() + (immediate ? 0 : 1), NULL, 10);
    result = new PTXNeg(arg1, arg2, immediate, line_num);
    return true;
  }
  return false;
}

PTXMul::PTXMul(int zero, int one, int two, bool imm, int line_num)
  : PTXInstruction(PTX_MULTIPLY, line_num), immediate(imm)
{
  args[0] = zero;
  args[1] = one;
  args[2] = two;
}

/*static*/
bool PTXMul::interpret(const std::string &line, int line_num,
                       PTXInstruction *&result)
{
  if (line.find("mul.") != std::string::npos)
  {
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    assert(tokens.size() == 4);
    int arg1 = strtol(tokens[1].c_str()+1, NULL, 10);
    int arg2 = strtol(tokens[2].c_str()+1, NULL, 10);
    const size_t regs = count(line, "%");
    assert((regs == 2) || (regs == 3));
    const bool immediate = (regs == 2);
    int arg3 = strtol(tokens[3].c_str() + (immediate ? 0 : 1), NULL, 10);
    result = new PTXMul(arg1, arg2, arg3, immediate, line_num);
    return true;
  }
  return false;
}

PTXMad::PTXMad(int a[4], bool imm[4], int line_num)
  : PTXInstruction(PTX_MAD, line_num)
{
  for (int i = 0; i < 4; i++)
    args[i] = a[i];
  for (int i = 0; i < 4; i++)
    immediate[i] = imm[i];
}

/*static*/
bool PTXMad::interpret(const std::string &line, int line_num,
                       PTXInstruction *&result)
{
  if (line.find("mad.") != std::string::npos)
  {
    if ((line.find("lo") != std::string::npos) ||
        (line.find("wide") != std::string::npos))
    {
      std::vector<std::string> tokens; 
      split(tokens, line.c_str());
      assert(tokens.size() == 5);
      int args[4];
      bool immediate[4];
      for (int i = 0; i < 4; i++)
      {
        const bool imm = (tokens[i+1].find("%") != std::string::npos);
        immediate[i] = imm;
        args[i] = strtol(tokens[i+1].c_str() + (imm ? 0 : 1), NULL, 10);
      }
      result = new PTXMad(args, immediate, line_num);
    }
    else
      assert(false); // TODO: implement hi
    return true;
  }
  return false;
}

PTXSetPred::PTXSetPred(int zero, int one, int two, 
                       bool imm, CompType comp, int line_num)
  : PTXInstruction(PTX_SET_PREDICATE, line_num), immediate(imm)
{
  args[0] = zero;
  args[1] = one;
  args[2] = two;
}

/*static*/
bool PTXSetPred::interpret(const std::string &line, int line_num,
                           PTXInstruction *&result)
{
  if (line.find("setp.") != std::string::npos)
  {
    std::vector<std::string> tokens;
    const size_t regs = count(line, "%");
    assert((regs == 2) || (regs == 3));
    split(tokens, line.c_str());
    assert(tokens.size() == 4);
    std::string predicate = tokens[1].substr(1, tokens[1].size() - 2);
    int arg1 = convert_predicate(predicate);
    int arg2 = strtol(tokens[2].c_str()+1, NULL, 10);
    const bool immediate = (regs == 2);
    int arg3 = strtol(tokens[3].c_str() + (immediate ? 0 : 1), NULL, 10);
    CompType comparison;
    if (line.find(".gt") != std::string::npos)
      comparison = COMP_GT;
    else if (line.find(".ge") != std::string::npos)
      comparison = COMP_GE;
    else if (line.find(".eq") != std::string::npos)
      comparison = COMP_EQ;
    else if (line.find(".ne") != std::string::npos)
      comparison = COMP_NE;
    else if (line.find(".le") != std::string::npos)
      comparison = COMP_LE;
    else if (line.find(".lt") != std::string::npos)
      comparison = COMP_LT;
    else
      assert(false);
    result = new PTXSetPred(arg1, arg2, arg3, 
                            immediate, comparison, line_num);
    return true;
  }
  return false;
}

PTXSelectPred::PTXSelectPred(int zero, int one, int two, int three,
                             bool two_imm, bool three_imm, int line_num)
  : PTXInstruction(PTX_SELECT_PREDICATE, line_num)
{
  predicate = three;
  args[0] = zero;
  args[1] = one;
  args[2] = two;
  immediate[0] = two_imm;
  immediate[1] = three_imm;
}

/*static*/
bool PTXSelectPred::interpret(const std::string &line, int line_num,
                              PTXInstruction *&result)
{
  if (line.find("selp.") != std::string::npos)
  {
    std::vector<std::string> tokens; 
    split(tokens, line.c_str());
    assert(tokens.size() == 5);
    int arg1 = strtol(tokens[1].c_str()+1, NULL, 10);
    bool two_imm = (tokens[2].find("%") == std::string::npos);
    int arg2 = strtol(tokens[2].c_str() + (two_imm ? 0 : 1), NULL, 10);
    bool three_imm = (tokens[3].find("%") == std::string::npos);
    int arg3 = strtol(tokens[3].c_str() + (three_imm ? 0 : 1), NULL, 10);
    std::string predicate = tokens[4].substr(1, tokens[4].size() - 2);
    int arg4 = convert_predicate(predicate);
    result = new PTXSelectPred(arg1, arg2, arg3, arg4,
                               two_imm, three_imm, line_num);
    return true;
  }
  return false;
}

PTXBarrier::PTXBarrier(int n, int c, bool s, int line_num)
  : PTXInstruction(PTX_BARRIER, line_num), name(n), count(c), sync(s)
{
}

void PTXBarrier::update_count(unsigned arrival_count)
{
  // If we didn't have a count before, set it to the full CTA width
  if (count < 0)
    count = arrival_count;
}

/*static*/
bool PTXBarrier::interpret(const std::string &line, int line_num,
                           PTXInstruction *&result)
{
  if ((line.find("bar.") != std::string::npos) &&
      (line.find("membar.") == std::string::npos))
  {
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    assert((tokens.size() == 2) || (tokens.size() == 3));
    int name = strtol(tokens[1].c_str(), NULL, 10);
    int count = -1;
    if (tokens.size() == 3)
      count = strtol(tokens[2].c_str(), NULL, 10);
    bool sync = (line.find("arrive") == std::string::npos);
    result = new PTXBarrier(name, count, sync, line_num);
    return true;
  }
  return false;
}

PTXSharedAccess::PTXSharedAccess(int a, int o, bool w, int line_num)
  : PTXInstruction(PTX_SHARED_ACCESS, line_num), 
    addr(a), offset(o), write(w)
{
}

/*static*/
bool PTXSharedAccess::interpret(const std::string &line, int line_num,
                                PTXInstruction *&result)
{
  if ((line.find(".shared.") != std::string::npos) &&
      (line.find(".align.") == std::string::npos))
  {
    int offset = 0;   
    int start_reg = line.find("[") + 1;
    int end_reg = line.find("+");
    int addr = strtol(line.c_str()+start_reg+1, NULL, 10);
    if (end_reg != (int) std::string::npos)
      offset = strtol(line.c_str()+end_reg+1, NULL, 10);
    bool write = (line.find("st.") != std::string::npos);
    result = new PTXSharedAccess(addr, offset, write, line_num);
    return true;
  }
  return false;
}

PTXConvert::PTXConvert(int zero, int one, int line_num)
  : PTXInstruction(PTX_CONVERT, line_num), src(one), dst(zero)
{
}

/*static*/
bool PTXConvert::interpret(const std::string &line, int line_num,
                           PTXInstruction *&result)
{
  if (line.find("cvt.") != std::string::npos)
  {
    std::vector<std::string> tokens; 
    split(tokens, line.c_str());
    assert(tokens.size() == 3);
    int arg1 = strtol(tokens[1].c_str()+1, NULL, 10);
    int arg2 = strtol(tokens[2].c_str()+1, NULL, 10);
    result = new PTXConvert(arg1, arg2, line_num);
    return true;
  }
  return false;
}

