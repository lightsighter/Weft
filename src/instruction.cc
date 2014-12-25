
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
  if (PTXSharedDecl::interpret(line, line_num, result))
    return result; 
  if (PTXMove::interpret(line, line_num, result))
    return result;
  if (PTXRightShift::interpret(line, line_num, result))
    return result;
  if (PTXLeftShift::interpret(line, line_num, result))
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

PTXLabel::PTXLabel(const std::string &line, int line_num)
  : PTXInstruction(PTX_LABEL, line_num)
{
}

void PTXLabel::update_labels(std::map<std::string,PTXInstruction*> &labels)
{

}

PTXBranch::PTXBranch(const std::string &line, int line_num)
  : PTXInstruction(PTX_BRANCH, line_num)
{
}

void PTXBranch::set_targets(const std::map<std::string,PTXInstruction*> &lablels)
{

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

PTXMove::PTXMove(const std::string &d, const std::string &s, int line_num)
  : PTXInstruction(PTX_MOVE, line_num), dst(d), src(s), immediate(false)
{
}

PTXMove::PTXMove(const std::string &d, int imm, int line_num)
  : PTXInstruction(PTX_MOVE, line_num), dst(d), immediate(true), 
    immediate_value(imm)
{
}

/*static*/
bool PTXMove::interpret(const std::string &line, int line_num,
                        PTXInstruction *&result)
{
  if (line.find("mov") != std::string::npos)
  {
    if (count(line, "%") == 2)
    {
      int start_arg1 = line.find("%");
      int end_arg1 = line.find(",");
      int start_arg2 = line.find("%", end_arg1);
      int end_arg2 = line.find(";");
      std::string arg1 = line.substr(start_arg1, end_arg1 - start_arg1);
      std::string arg2 = line.substr(start_arg2, end_arg2 - start_arg2);
      result = new PTXMove(arg1, arg2, line_num);
      return true;
    }
    else if (contains(line, "cuda"))
    {
      int start_arg1 = line.find("%");
      int end_arg1 = line.find(",");
      int start_arg2 = line.find("_", end_arg1);
      int end_arg2 = line.find(";");
      std::string arg1 = line.substr(start_arg1, end_arg1 - start_arg1);
      std::string arg2 = line.substr(start_arg2, end_arg2 - start_arg2);
      result = new PTXMove(arg1, arg2, line_num);
      return true;
    }
    else if (count(line, "%") == 1)
    {
      std::vector<std::string> tokens;
      split(tokens, line.c_str()); 
      assert(tokens.size() == 3);
      std::string arg1 = tokens[1].substr(0, tokens[1].size()-1);
      int arg2 = atoi(tokens[2].substr(0, tokens[2].size()-1).c_str());
      result = new PTXMove(arg1, arg2, line_num);
      return true;
    }
  }
  return false;
}

PTXRightShift::PTXRightShift(const std::string &d, const std::string &s,
                             int shift, int line_num)
 : PTXInstruction(PTX_RIGHT_SHIFT, line_num), 
   dst(d), src(s), shift_value(shift)
{
}

/*static*/
bool PTXRightShift::interpret(const std::string &line, int line_num,
                              PTXInstruction *&result)
{
  if (line.find("shr.") != std::string::npos)
  {
    assert(count(line, "%") == 2);
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    assert(tokens.size() == 4);
    std::string arg1 = tokens[1].substr(0, tokens[1].size() - 1);
    std::string arg2 = tokens[2].substr(0, tokens[2].size() - 1);
    int arg3 = atoi(tokens[3].substr(0, tokens[3].size() - 1).c_str());
    result = new PTXRightShift(arg1, arg2, arg3, line_num);
    return true;
  }
  return false;
}

PTXLeftShift::PTXLeftShift(const std::string &d, const std::string &s,
                             int shift, int line_num)
 : PTXInstruction(PTX_LEFT_SHIFT, line_num), 
   dst(d), src(s), shift_value(shift)
{
}

/*static*/
bool PTXLeftShift::interpret(const std::string &line, int line_num,
                              PTXInstruction *&result)
{
  if (line.find("shl.") != std::string::npos)
  {
    assert(count(line, "%") == 2);
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    assert(tokens.size() == 4);
    std::string arg1 = tokens[1].substr(0, tokens[1].size() - 1);
    std::string arg2 = tokens[2].substr(0, tokens[2].size() - 1);
    int arg3 = atoi(tokens[3].substr(0, tokens[3].size() - 1).c_str());
    result = new PTXLeftShift(arg1, arg2, arg3, line_num);
    return true;
  }
  return false;
}


