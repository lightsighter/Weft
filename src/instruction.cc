
#include "race.h"
#include "program.h"
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

inline bool filter_identifier(char c)
{
  if (c == '_')
    return true;
  if (c == '$')
    return true;
  if (('0' <= c) && (c <= '9'))
    return true;
  if (('a' <= c) && (c <= 'z'))
    return true;
  if (('A' <= c) && (c <= 'Z'))
    return true;
  return false;
}

inline uint64_t convert_identifier(char c)
{
  if (('0' <= c) && (c <= '9'))
    return (c - '0');
  if (('a' <= c) && (c <= 'z'))
    return ((c - 'a') + 10);
  if (('A' <= c) && (c <= 'Z'))
    return ((c - 'A') + 36);
  if (c == '_')
    return 62;
  if (c == '$')
    return 63;
  assert(false);
  return -1;
}

inline char convert_identifier_back(uint64_t c)
{
  if (c < 10)
    return '0' + c;
  if (c < 36)
    return 'a' + (c - 10);
  if (c < 62)
    return 'A' + (c - 36);
  if (c == 62)
    return '_';
  if (c == 63)
    return '$';
  assert(false);
  return '\0';
}

inline int64_t compress_identifier(const char *buffer, size_t buffer_size)
{
  // There are 64 (2^6) valid character identifiers so we can't
  // encode more than 10 of them in a 64 bit integer
  assert(buffer_size <= 10);
  uint64_t result = 0;
  for (int i = 0; i < buffer_size; i++)
  {
    uint64_t next = convert_identifier(buffer[i]);
    result |= (next << (i*6));
  }
  // Encode the size in the upper 3 bits
  result |= (buffer_size << 60);
  return result;
}

inline void decompress_identifier(uint64_t id, char *buffer,
                                  size_t buffer_size)
{
  size_t id_size = ((7L << 60) & id) >> 60;
  assert(id_size <= 10);
  int min_size = ((buffer_size-1) < id_size) ? (buffer_size-1) : id_size;
  buffer[min_size] = '\0';
  for (int i = 0; i < min_size; i++)
  {
    uint64_t c = ((63L << (i*6)) & id) >> (i*6);
    buffer[i] = convert_identifier_back(c);
  }
}

inline int64_t parse_predicate(const std::string &pred, bool &negate)
{
  negate = false;
  const char *str = pred.c_str();
  size_t size = pred.size();
  unsigned index = 0;
  while (index < size)
  {
    if (filter_identifier(str[index]))
      break;
    if (str[index] == '!')
      negate = true;
    index++;
  }
  assert(index < size);
  char buffer[10];
  unsigned buffer_size = 0;
  while ((index < size) && (buffer_size < 10))
  {
    if (!filter_identifier(str[index]))
      break;
    buffer[buffer_size] = str[index];
    buffer_size++;
    index++;
  }
  return compress_identifier(buffer, buffer_size);
}

inline int64_t parse_register(const std::string &reg)
{
  // Do a quick check for any of the special registers
  if (reg.find("%tid") != std::string::npos)
    return WEFT_TID_REG;
  if (reg.find("%ctaid") != std::string::npos)
    return WEFT_CTA_REG;
  // Find the start of the identifier
  const char *str = reg.c_str();
  size_t size = reg.size();
  unsigned index = 0;
  while (index < size)
  {
    if (filter_identifier(str[index]))
      break;
    index++;
  }
  // If we read through the whole string that is bad
  assert(index < size);
  char buffer[10];
  unsigned buffer_size = 0;
  while ((index < size) && (buffer_size < 10))
  {
    if (!filter_identifier(str[index]))
      break;
    buffer[buffer_size] = str[index];
    buffer_size++;
    index++;
  }
  return compress_identifier(buffer, buffer_size);
}

inline int64_t parse_immediate(const std::string &imm)
{
  return strtol(imm.c_str(), NULL, 10);
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

/*static*/
uint64_t PTXInstruction::compress_identifier(const char *buffer, 
                                             size_t buffer_size)
{
  return ::compress_identifier(buffer, buffer_size);
}

/*static*/
void PTXInstruction::decompress_identifier(uint64_t id, char *buffer,
                                           size_t buffer_size)
{
  ::decompress_identifier(id, buffer, buffer_size);
}

PTXLabel::PTXLabel(const std::string &l, int line_num)
  : PTXInstruction(PTX_LABEL, line_num), label(l)
{
}

PTXInstruction* PTXLabel::emulate(Thread *thread)
{
  return next;
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

PTXBranch::PTXBranch(int64_t p, bool n, const std::string &l, int line_num)
  : PTXInstruction(PTX_BRANCH, line_num), 
    predicate(p), negate(n), label(l), target(NULL)
{
}

PTXInstruction* PTXBranch::emulate(Thread *thread)
{
  // Handle the uniform branch case
  if (predicate == 0)
    return target;
  bool value;
  if (!thread->get_pred(predicate, value))
  {
#if 0
    char buffer[11];
    decompress_identifier(predicate, buffer, 11);
    fprintf(stderr,"WEFT WARNING: Branch depends on undefined predicate %s\n",
                   buffer);
#endif
    return next;
  }
  if (negate)
    value = !value;
  if (value)
    return target;
  return next;
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
      bool negate;
      int64_t predicate = parse_predicate(tokens[0], negate);
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

PTXSharedDecl::PTXSharedDecl(const std::string &n, int64_t addr, int line_num)
  : PTXInstruction(PTX_SHARED_DECL, line_num), name(n), address(addr)
{
}

PTXInstruction* PTXSharedDecl::emulate(Thread *thread)
{
  thread->register_shared_location(name, address);
  return next;
}

/*static*/
bool PTXSharedDecl::interpret(const std::string &line, int line_num,
                              PTXInstruction *&result)
{
  static int64_t shared_offset = 1;
  if ((line.find(".shared") != std::string::npos) &&
      (line.find(".align") != std::string::npos))
  {
    int start = line.find("_");
    std::string name = line.substr(start, line.find("[") - start);
    // This is just an approximation to stride all 
    // the shared memory allocations far away from each other
    int64_t  address = shared_offset * SDDRINC;
    shared_offset++;
    result = new PTXSharedDecl(name, address, line_num);
    return true;
  }
  return false;
}

PTXMove::PTXMove(int64_t dst, int64_t src, bool imm, int line_num)
  : PTXInstruction(PTX_MOVE, line_num), immediate(imm)
{
  args[0] = dst;
  args[1] = src;
}

PTXMove::PTXMove(int64_t dst, const std::string &src, int line_num)
  : PTXInstruction(PTX_MOVE, line_num), immediate(false)
{
  args[0] = dst;
  source = src;
}

PTXInstruction* PTXMove::emulate(Thread *thread)
{
  if (!source.empty())
  {
    int64_t addr;
    if (!thread->find_shared_location(source, addr))
      return next;
    thread->set_value(args[0], addr);
  }
  else if (immediate)
  {
    thread->set_value(args[0], args[1]);
  }
  else
  {
    int64_t value;
    if (!thread->get_value(args[1], value))
      return next;
    thread->set_value(args[0], value);
  }
  return next;
}

/*static*/
bool PTXMove::interpret(const std::string &line, int line_num,
                        PTXInstruction *&result)
{
  if (line.find("mov.") != std::string::npos)
  {
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    if (tokens.size() != 3)
      return false;
    assert(tokens.size() == 3);
    if (count(line, "%") == 2)
    {
      int64_t arg1 = parse_register(tokens[1]);
      int64_t arg2 = parse_register(tokens[2]);
      result = new PTXMove(arg1, arg2, false/*immediate*/, line_num);
      return true;
    }
    else if (contains(line, "cuda"))
    {
      int end_arg1 = line.find(",");
      int start_arg2 = line.find("_", end_arg1);
      int end_arg2 = line.find(";");
      int64_t arg1 = parse_register(tokens[1]);
      std::string arg2 = line.substr(start_arg2, end_arg2 - start_arg2);
      result = new PTXMove(arg1, arg2, line_num);
      return true;
    }
    else if (count(line, "%") == 1)
    {
      int64_t arg1 = parse_register(tokens[1]);
      int64_t arg2 = parse_immediate(tokens[2]);
      result = new PTXMove(arg1, arg2, true/*immediate*/, line_num);
      return true;
    }
  }
  return false;
}

PTXRightShift::PTXRightShift(int64_t one, int64_t two, int64_t three, 
                             bool imm, int line_num)
 : PTXInstruction(PTX_RIGHT_SHIFT, line_num), immediate(imm)
{
  args[0] = one;
  args[1] = two;
  args[2] = three;
}

PTXInstruction* PTXRightShift::emulate(Thread *thread)
{
  if (immediate)
  {
    int64_t source;
    if (!thread->get_value(args[1], source))
      return next;
    int64_t value = source >> args[2];
    thread->set_value(args[0], value);
  }
  else
  {
    int64_t source, shift;
    if (!thread->get_value(args[1], source))
      return next;
    if (!thread->get_value(args[2], shift))
      return next;
    int64_t value = source >> shift;
    thread->set_value(args[0], value);
  }
  return next;
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
    int64_t arg1 = parse_register(tokens[1]);
    int64_t arg2 = parse_register(tokens[2]);
    const size_t regs = count(line, "%");
    assert((regs == 2) || (regs == 3));
    const bool immediate = (regs == 2);
    int64_t arg3 = immediate ? parse_immediate(tokens[3]) 
                             : parse_register(tokens[3]);
    result = new PTXRightShift(arg1, arg2, arg3, immediate, line_num);
    return true;
  }
  return false;
}

PTXLeftShift::PTXLeftShift(int64_t one, int64_t two, int64_t three, 
                             bool imm, int line_num)
 : PTXInstruction(PTX_LEFT_SHIFT, line_num), immediate(imm)
{
  args[0] = one;
  args[1] = two;
  args[2] = three;
}

PTXInstruction* PTXLeftShift::emulate(Thread *thread)
{
  if (immediate)
  {
    int64_t source;
    if (!thread->get_value(args[1], source))
      return next;
    int64_t value = source << args[2];
    thread->set_value(args[0], value);
  }
  else
  {
    int64_t source, shift;
    if (!thread->get_value(args[1], source))
      return next;
    if (!thread->get_value(args[2], shift))
      return next;
    int64_t value = source << shift;
    thread->set_value(args[0], value);
  }
  return next;
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
    int64_t arg1 = parse_register(tokens[1]);
    int64_t arg2 = parse_register(tokens[2]);
    const size_t regs = count(line, "%");
    assert((regs == 2) || (regs == 3));
    const bool immediate = (regs == 2);
    int64_t arg3 = immediate ? parse_immediate(tokens[3])
                             : parse_register(tokens[3]);
    result = new PTXLeftShift(arg1, arg2, arg3, immediate, line_num);
    return true;
  }
  return false;
}

PTXAnd::PTXAnd(int64_t zero, int64_t one, int64_t two, 
               bool imm, int line_num)
  : PTXInstruction(PTX_AND, line_num), immediate(imm)
{
  args[0] = zero;
  args[1] = one;
  args[2] = two;
}

PTXInstruction* PTXAnd::emulate(Thread *thread)
{
  if (immediate)
  {
    int64_t source;
    if (!thread->get_value(args[1], source))
      return next;
    int64_t value = source & args[2];
    thread->set_value(args[0], value);
  }
  else
  {
    int64_t source, other;
    if (!thread->get_value(args[1], source))
      return next;
    if (!thread->get_value(args[2], other))
      return next;
    int64_t value = source & other;
    thread->set_value(args[0], value);
  }
  return next;
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
    int64_t arg1 = parse_register(tokens[1]);
    int64_t arg2 = parse_register(tokens[2]);
    const size_t regs = count(line, "%");
    assert((regs == 2) || (regs == 3));
    const bool immediate = (regs == 2);
    int64_t arg3 = immediate ? parse_immediate(tokens[3])
                             : parse_register(tokens[3]);
    result = new PTXAnd(arg1, arg2, arg3, immediate, line_num);
    return true;
  }
  return false;
}

PTXOr::PTXOr(int64_t zero, int64_t one, int64_t two, 
             bool imm, int line_num)
  : PTXInstruction(PTX_OR, line_num), immediate(imm)
{
  args[0] = zero;
  args[1] = one;
  args[2] = two;
}

PTXInstruction* PTXOr::emulate(Thread *thread)
{
  if (immediate)
  {
    int64_t source;
    if (!thread->get_value(args[1], source))
      return next;
    int64_t value = source | args[2];
    thread->set_value(args[0], value);
  }
  else
  {
    int64_t source, other;
    if (!thread->get_value(args[1], source))
      return next;
    if (!thread->get_value(args[2], other))
      return next;
    int64_t value = source | other;
    thread->set_value(args[0], value);
  }
  return next;
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
    int64_t arg1 = parse_register(tokens[1]);
    int64_t arg2 = parse_register(tokens[2]);
    const size_t regs = count(line, "%");
    assert((regs == 2) || (regs == 3));
    const bool immediate = (regs == 2);
    int64_t arg3 = immediate ? parse_immediate(tokens[3])
                             : parse_register(tokens[3]);
    result = new PTXOr(arg1, arg2, arg3, immediate, line_num);
    return true;
  }
  return false;
}

PTXAdd::PTXAdd(int64_t zero, int64_t one, int64_t two, 
               bool imm, int line_num)
  : PTXInstruction(PTX_ADD, line_num), immediate(imm)
{
  args[0] = zero;
  args[1] = one;
  args[2] = two;
}

PTXInstruction* PTXAdd::emulate(Thread *thread)
{
  if (immediate)
  {
    int64_t source;
    if (!thread->get_value(args[1], source))
      return next;
    int64_t value = source + args[2];
    thread->set_value(args[0], value);
  }
  else
  {
    int64_t source, other;
    if (!thread->get_value(args[1], source))
      return next;
    if (!thread->get_value(args[2], other))
      return next;
    int64_t value = source + other;
    thread->set_value(args[0], value);
  }
  return next;
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
    int64_t arg1 = parse_register(tokens[1]);
    int64_t arg2 = parse_register(tokens[2]);
    const size_t regs = count(line, "%");
    assert((regs == 2) || (regs == 3));
    const bool immediate = (regs == 2);
    int64_t arg3 = immediate ? parse_immediate(tokens[3])
                             : parse_register(tokens[3]);
    result = new PTXAdd(arg1, arg2, arg3, immediate, line_num);
    return true;
  }
  return false;
}

PTXSub::PTXSub(int64_t zero, int64_t one, int64_t two, 
               bool imm, int line_num)
  : PTXInstruction(PTX_SUB, line_num), immediate(imm)
{
  args[0] = zero;
  args[1] = one;
  args[2] = two;
}

PTXInstruction* PTXSub::emulate(Thread *thread)
{
  if (immediate)
  {
    int64_t source;
    if (!thread->get_value(args[1], source))
      return next;
    int64_t value = source - args[2];
    thread->set_value(args[0], value);
  }
  else
  {
    int64_t source, other;
    if (!thread->get_value(args[1], source))
      return next;
    if (!thread->get_value(args[2], other))
      return next;
    int64_t value = source - other;
    thread->set_value(args[0], value);
  }
  return next;
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
    int64_t arg1 = parse_register(tokens[1]);
    int64_t arg2 = parse_register(tokens[2]);
    const size_t regs = count(line, "%");
    assert((regs == 2) || (regs == 3));
    const bool immediate = (regs == 2);
    int64_t arg3 = immediate ? parse_immediate(tokens[3])
                             : parse_register(tokens[3]);
    result = new PTXSub(arg1, arg2, arg3, immediate, line_num);
    return true;
  }
  return false;
}

PTXNeg::PTXNeg(int64_t zero, int64_t one, bool imm, int line_num)
  : PTXInstruction(PTX_NEGATE, line_num), immediate(imm)
{
  args[0] = zero;
  args[1] = one;
}

PTXInstruction* PTXNeg::emulate(Thread *thread)
{
  if (immediate)
  {
    thread->set_value(args[0], ~args[1]);
  }
  else
  {
    int64_t source;
    if (!thread->get_value(args[1], source))
      return next;
    thread->set_value(args[0], ~source);
  }
  return next;
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
    int64_t arg1 = parse_register(tokens[1]);
    const size_t regs = count(line, "%");
    assert((regs == 1) || (regs == 2));
    const bool immediate = (regs == 1);
    int64_t arg2 = immediate ? parse_immediate(tokens[2])
                             : parse_register(tokens[2]);
    result = new PTXNeg(arg1, arg2, immediate, line_num);
    return true;
  }
  return false;
}

PTXMul::PTXMul(int64_t zero, int64_t one, int64_t two, bool imm, int line_num)
  : PTXInstruction(PTX_MULTIPLY, line_num), immediate(imm)
{
  args[0] = zero;
  args[1] = one;
  args[2] = two;
}

PTXInstruction* PTXMul::emulate(Thread *thread)
{
  if (immediate)
  {
    int64_t source;
    if (!thread->get_value(args[1], source))
      return next;
    int64_t value = source * args[2];
    thread->set_value(args[0], value);
  }
  else
  {
    int64_t source, other;
    if (!thread->get_value(args[1], source))
      return next;
    if (!thread->get_value(args[2], other))
      return next;
    int64_t value = source * other;
    thread->set_value(args[0], value);
  }
  return next;
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
    int64_t arg1 = parse_register(tokens[1]);
    int64_t arg2 = parse_register(tokens[2]);
    const size_t regs = count(line, "%");
    assert((regs == 2) || (regs == 3));
    const bool immediate = (regs == 2);
    int64_t arg3 = immediate ? parse_immediate(tokens[3])
                             : parse_register(tokens[3]);
    result = new PTXMul(arg1, arg2, arg3, immediate, line_num);
    return true;
  }
  return false;
}

PTXMad::PTXMad(int64_t a[4], bool imm[4], int line_num)
  : PTXInstruction(PTX_MAD, line_num)
{
  for (int i = 0; i < 4; i++)
    args[i] = a[i];
  for (int i = 0; i < 4; i++)
    immediate[i] = imm[i];
}

PTXInstruction* PTXMad::emulate(Thread *thread)
{
  int64_t vals[3];
  for (int i = 0; i < 3; i++)
  {
    if (immediate[i+1])
      vals[i] = args[i+1];
    else if (!thread->get_value(args[i+1], vals[i]))
      return next;
  }
  int64_t value = vals[0] * vals[1] + vals[2];
  thread->set_value(args[0], value);
  return next;
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
      int64_t args[4];
      bool immediate[4];
      for (int i = 0; i < 4; i++)
      {
        const bool imm = (tokens[i+1].find("%") == std::string::npos);
        immediate[i] = imm;
        args[i] = imm ? parse_immediate(tokens[i+1])
                      : parse_register(tokens[i+1]);
      }
      result = new PTXMad(args, immediate, line_num);
    }
    else
      assert(false); // TODO: implement hi
    return true;
  }
  return false;
}

PTXSetPred::PTXSetPred(int64_t zero, int64_t one, int64_t two, 
                       bool imm, CompType comp, int line_num)
  : PTXInstruction(PTX_SET_PREDICATE, line_num), 
    comparison(comp), immediate(imm)
{
  args[0] = zero;
  args[1] = one;
  args[2] = two;
}

PTXInstruction* PTXSetPred::emulate(Thread *thread)
{
  int64_t vals[2];
  if (!thread->get_value(args[1], vals[0]))
    return next;
  if (immediate)
    vals[1] = args[2];
  else if (!thread->get_value(args[2], vals[1]))
    return next;
  bool value;
  switch (comparison)
  {
    case COMP_GT:
      {
        value = (vals[0] > vals[1]);
        break;
      }
    case COMP_GE:
      {
        value = (vals[0] >= vals[1]);
        break;
      }
    case COMP_EQ:
      {
        value = (vals[0] == vals[1]);
        break;
      }
    case COMP_NE:
      {
        value = (vals[0] != vals[1]);
        break;
      }
    case COMP_LE:
      {
        value = (vals[0] <= vals[1]);
        break;
      }
    case COMP_LT:
      {
        value = (vals[0] < vals[1]);
        break;
      }
    default:
      assert(false);
  }
  thread->set_pred(args[0], value);
  return next;
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
    bool negate;
    int64_t arg1 = parse_predicate(tokens[1], negate);
    assert(!negate);
    int64_t arg2 = parse_register(tokens[2]);
    const bool immediate = (regs == 2);
    int64_t arg3 = immediate ? parse_immediate(tokens[3])
                             : parse_register(tokens[3]);
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

PTXSelectPred::PTXSelectPred(int64_t zero, int64_t one, int64_t two, int64_t three,
                             bool neg, bool two_imm, bool three_imm, int line_num)
  : PTXInstruction(PTX_SELECT_PREDICATE, line_num)
{
  negate = neg;
  predicate = three;
  args[0] = zero;
  args[1] = one;
  args[2] = two;
  immediate[0] = two_imm;
  immediate[1] = three_imm;
}

PTXInstruction* PTXSelectPred::emulate(Thread *thread)
{
  int64_t vals[2];
  for (int i = 0; i < 2; i++)
  {
    if (immediate[i])
      vals[i] = args[i+1];
    else if (!thread->get_value(args[i+1], vals[i]))
      return next;
  }
  bool pred;
  if (!thread->get_pred(predicate, pred))
    return next;
  if (negate)
    pred = !pred;
  int64_t value = (pred ? vals[0] : vals[1]);
  thread->set_value(args[0], value);
  return next;
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
    int64_t arg1 = parse_register(tokens[1]);
    bool two_imm = (tokens[2].find("%") == std::string::npos);
    int64_t arg2 = two_imm ? parse_immediate(tokens[2])
                       : parse_register(tokens[2]);
    bool three_imm = (tokens[3].find("%") == std::string::npos);
    int64_t arg3 = three_imm ? parse_immediate(tokens[3])
                         : parse_register(tokens[3]);
    bool negate;
    int64_t arg4 = parse_predicate(tokens[4], negate);
    result = new PTXSelectPred(arg1, arg2, arg3, arg4, negate,
                               two_imm, three_imm, line_num);
    return true;
  }
  return false;
}

PTXBarrier::PTXBarrier(int64_t n, int64_t c, bool s, 
                       bool name_imm, bool count_imm, int line_num)
  : PTXInstruction(PTX_BARRIER, line_num), name(n), 
    count(c), sync(s), name_immediate(name_imm), count_immediate(count_imm)
{
}

PTXInstruction* PTXBarrier::emulate(Thread *thread)
{
  WeftInstruction *instruction;
  int64_t name_value;
  if (!name_immediate)
  {
    if (!thread->get_value(name, name_value))
      return next;
  }
  else
    name_value = name;
  int64_t count_value;
  if (!count_immediate)
  {
    if (!thread->get_value(count, count_value))
      return next;
  }
  else
    count_value = count;
  if (sync)
    instruction = new BarrierSync(name_value, count_value, this, thread);
  else
    instruction = new BarrierArrive(name_value, count_value, this, thread);
  thread->add_instruction(instruction);
  thread->update_max_barrier_name(name_value);
  return next;
}

void PTXBarrier::update_count(unsigned arrival_count)
{
  // If we didn't have a count before, set it to the full CTA width
  if (count < 0)
  {
    count = arrival_count;
    count_immediate = true;
  }
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
    bool name_immediate = false;
    int64_t name;
    if (tokens[1].find("%") == std::string::npos)
    {
      name = parse_immediate(tokens[1]);
      name_immediate = true;
    }
    else
      name = parse_register(tokens[1]);
    int64_t count = -1;
    bool count_immediate = false;
    if (tokens.size() == 3)
    {
      if (tokens[2].find("%") == std::string::npos)
      {
        count = parse_immediate(tokens[2]);
        count_immediate = true;
      }
      else
        count = parse_register(tokens[2]);
    }
    bool sync = (line.find("arrive") == std::string::npos);
    result = new PTXBarrier(name, count, sync, name_immediate, 
                            count_immediate, line_num);
    return true;
  }
  return false;
}

PTXSharedAccess::PTXSharedAccess(int64_t a, int64_t o, bool w, int line_num)
  : PTXInstruction(PTX_SHARED_ACCESS, line_num), 
    addr(a), offset(o), write(w)
{
}

PTXInstruction* PTXSharedAccess::emulate(Thread *thread)
{
  int64_t value;
  if (!thread->get_value(addr, value))
    return next;
  int64_t address = value + offset;
  WeftInstruction *instruction;
  if (write)
    instruction = new SharedWrite(address, this, thread);
  else
    instruction = new SharedRead(address, this, thread);
  thread->add_instruction(instruction);
  return next;
}

/*static*/
bool PTXSharedAccess::interpret(const std::string &line, int line_num,
                                PTXInstruction *&result)
{
  if ((line.find(".shared.") != std::string::npos) &&
      (line.find(".align.") == std::string::npos))
  {
    int64_t offset = 0;   
    int start_reg = line.find("[") + 1;
    int end_reg = line.find("+")+1;
    int64_t addr = parse_register(line.substr(start_reg));
    if (end_reg != (int) std::string::npos)
      offset = parse_immediate(line.substr(end_reg));
    bool write = (line.find("st.") != std::string::npos);
    result = new PTXSharedAccess(addr, offset, write, line_num);
    return true;
  }
  return false;
}

PTXConvert::PTXConvert(int64_t zero, int64_t one, int line_num)
  : PTXInstruction(PTX_CONVERT, line_num), src(one), dst(zero)
{
}

PTXInstruction* PTXConvert::emulate(Thread *thread)
{
  int64_t value;
  if (!thread->get_value(src, value))
    return next;
  thread->set_value(dst, value);
  return next;
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
    int64_t arg1 = parse_register(tokens[1]);
    int64_t arg2 = parse_register(tokens[2]);
    result = new PTXConvert(arg1, arg2, line_num);
    return true;
  }
  return false;
}

PTXConvertAddress::PTXConvertAddress(int64_t zero, int64_t one, int line_num)
  : PTXInstruction(PTX_CONVERT_ADDRESS, line_num), src(one), dst(zero)
{
}

PTXInstruction* PTXConvertAddress::emulate(Thread *thread)
{
  int64_t value;
  if (!thread->get_value(src, value))
    return next;
  thread->set_value(dst, value);
  return next;
}

/*static*/
bool PTXConvertAddress::interpret(const std::string &line, int line_num,
                                  PTXInstruction *&result)
{
  if (line.find("cvta.") != std::string::npos)
  {
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    assert(tokens.size() == 3);
    int64_t arg1 = parse_register(tokens[1]);
    int64_t arg2 = parse_register(tokens[2]);
    result = new PTXConvertAddress(arg1, arg2, line_num);
    return true;
  }
  return false;
}

WeftInstruction::WeftInstruction(PTXInstruction *inst, Thread *t)
  : instruction(inst), thread(t), thread_line_number(t->get_program_size()),
    happens_relationship(NULL)
{
}

void WeftInstruction::initialize_happens(Happens *happens)
{
  assert(happens != NULL);
  assert(happens_relationship == NULL);
  happens_relationship = happens;
}

WeftBarrier::WeftBarrier(int n, int c, PTXBarrier *bar, Thread *thread)
  : WeftInstruction(bar, thread), name(n), count(c), 
    barrier(bar), instance(NULL)
{
}

void WeftBarrier::set_instance(BarrierInstance *inst)
{
  assert(instance == NULL);
  assert(inst != NULL);
  instance = inst;
}

BarrierSync::BarrierSync(int n, int c, PTXBarrier *bar, Thread *thread)
  : WeftBarrier(n, c, bar, thread)
{
}

BarrierArrive::BarrierArrive(int n, int c, PTXBarrier *bar, Thread *thread)
  : WeftBarrier(n, c, bar, thread)
{
}

WeftAccess::WeftAccess(int addr, PTXSharedAccess *acc, Thread *thread)
  : WeftInstruction(acc, thread), address(addr), access(acc)
{
}

SharedWrite::SharedWrite(int addr, PTXSharedAccess *acc, Thread *thread)
  : WeftAccess(addr, acc, thread)
{
}

SharedRead::SharedRead(int addr, PTXSharedAccess *acc, Thread *thread)
  : WeftAccess(addr, acc, thread)
{
}

