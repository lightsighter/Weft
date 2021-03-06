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

#include "race.h"
#include "weft.h"
#include "program.h"
#include "instruction.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

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
  for (unsigned i = 0; i < buffer_size; i++)
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
  if (reg.find("%tid.x") != std::string::npos)
    return WEFT_TID_X_REG;
  if (reg.find("%tid.y") != std::string::npos)
    return WEFT_TID_Y_REG;
  if (reg.find("%tid.z") != std::string::npos)
    return WEFT_TID_Z_REG;
  if (reg.find("%ntid.x") != std::string::npos)
    return WEFT_NTID_X_REG;
  if (reg.find("%ntid.y") != std::string::npos)
    return WEFT_NTID_Y_REG;
  if (reg.find("%ntid.z") != std::string::npos)
    return WEFT_NTID_Z_REG;
  if (reg.find("%laneid") != std::string::npos)
    return WEFT_LANE_REG;
  if (reg.find("%warpid") != std::string::npos)
    return WEFT_WARP_REG;
  if (reg.find("%nwarpid") != std::string::npos)
    return WEFT_NWARP_REG;
  if (reg.find("%ctaid.x") != std::string::npos)
    return WEFT_CTA_X_REG;
  if (reg.find("%ctaid.y") != std::string::npos)
    return WEFT_CTA_Y_REG;
  if (reg.find("%ctaid.z") != std::string::npos)
    return WEFT_CTA_Z_REG;
  if (reg.find("%nctaid.x") != std::string::npos)
    return WEFT_NCTA_X_REG;
  if (reg.find("%nctaid.y") != std::string::npos)
    return WEFT_NCTA_Y_REG;
  if (reg.find("%nctaid.z") != std::string::npos)
    return WEFT_NCTA_Z_REG;

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
  : kind(PTX_LAST), line_number(0), next(NULL),
    source_file(NULL), source_line_number(-1)
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

PTXInstruction* PTXInstruction::emulate_warp(Thread **threads,
                                             ThreadState *thread_state,
                                             int &shared_access_id,
                                             SharedStore &store)
{
  // For most instructions, we can just call
  // emulate on individual threads that are enabled.
  // Instructions which are different can override this behavior.
  for (int i = 0; i < WARP_SIZE; i++)
  {
    if (thread_state[i].status == THREAD_ENABLED)
      emulate(threads[i]);
  }
  return next;
}

void PTXInstruction::set_next(PTXInstruction *n)
{
  assert(n != NULL);
  assert(next == NULL);
  next = n;
}

void PTXInstruction::set_source_location(const char *file, int line)
{
  source_file = file;
  source_line_number = line;
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
  if (PTXXor::interpret(line, line_num, result))
    return result;
  if (PTXNot::interpret(line, line_num, result))
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
  if (PTXBitFieldExtract::interpret(line, line_num, result))
    return result;
  if (PTXShuffle::interpret(line, line_num, result))
    return result;
  if (PTXExit::interpret(line, line_num, result))
    return result;
  if (PTXGlobalDecl::interpret(line, line_num, result))
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
    case PTX_XOR:
      return "Logical Xor";
    case PTX_NOT:
      return "Logical Not";
    case PTX_ADD:
      return "Integer Add";
    case PTX_SUB:
      return "Integer Subtract";
    case PTX_NEGATE:
      return "Negation";
    case PTX_CONVERT:
      return "Convert";
    case PTX_CONVERT_ADDRESS:
      return "Convert Address";
    case PTX_BFE:
      return "Bit Field Extract";
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
    case PTX_SHFL:
      return "Shuffle";
    case PTX_EXIT:
      return "Exit";
    case PTX_GLOBAL_DECL:
      return "Global Memory Declaration";
    case PTX_GLOBAL_LOAD:
      return "Global Load";
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

PTXInstruction* PTXLabel::emulate_warp(Thread **threads,
                                       ThreadState *thread_state,
                                       int &shared_access_id,
                                       SharedStore &store)
{
  // Always check for convergence at the start of basic blocks
  for (int i = 0; i < WARP_SIZE; i++)
  {
    if ((thread_state[i].status == THREAD_DISABLED) &&
        (thread_state[i].next == this))
    {
      thread_state[i].status = THREAD_ENABLED;
      thread_state[i].next = NULL;
    }
  }
  return next;
}

void PTXLabel::update_labels(std::map<std::string,PTXLabel*> &labels)
{
  std::map<std::string,PTXLabel*>::const_iterator finder = 
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
    if (thread->program->weft->report_warnings())
    {
      char buffer[11];
      decompress_identifier(predicate, buffer, 11);
      fprintf(stderr,"WEFT WARNING: Branch depends on undefined predicate %s\n",
                     buffer);
    }
    return next;
  }
  if (negate)
    value = !value;
  if (value)
    return target;
  return next;
}

PTXInstruction* PTXBranch::emulate_warp(Thread **threads,
                                        ThreadState *thread_state,
                                        int &shared_access_id,
                                        SharedStore &store)
{
  // Evaluate all the branches for all the enabled threads
  PTXInstruction *targets[WARP_SIZE];
  for (int i = 0; i < WARP_SIZE; i++)
  {
    if (thread_state[i].status == THREAD_ENABLED)
    {
      if (predicate == 0)
        targets[i] = target;
      else
      {
        bool value;
        if (!threads[i]->get_pred(predicate, value))
        {
          if (threads[i]->program->weft->report_warnings())
          {
            char buffer[11];
            decompress_identifier(predicate, buffer, 11);
            fprintf(stderr,"WEFT WARNING: Branch depends on "
                           "undefined predicate %s\n", buffer);
          }
          targets[i] = next;
        }
        else
        {
          // We successfully got a predicate
          if (negate)
            value = !value;
          if (value)
            targets[i] = target;
          else
            targets[i] = next;
        }
      }
    }
    else if (thread_state[i].status == THREAD_DISABLED)
    {
      targets[i] = thread_state[i].next;
      assert(targets[i] != NULL);
    }
    else
      targets[i] = NULL; // exitted threads don't matter
  }
  // We have all the targets 
  // See if we have consensus about where to go next 
  bool converged = true; 
  PTXInstruction *result = NULL;
  for (int i = 0; i < WARP_SIZE; i++)
  {
    // Skip any exitted threads
    if (targets[i] == NULL)
      continue;
    // If this is our first non-NULL entry record it
    if (result == NULL)
    {
      result = targets[i];
      continue;
    }
    // If we diverge at some point then we are done
    if (targets[i] != result)
    {
      converged = false;
      break;
    }
  }
  // If all the threads have exitted we should never be here
  assert(result != NULL);
  // If we converged, then we can all go to the same place
  if (converged)
  {
    // Enable all the non-exitted threads and return the result
    for (int i = 0; i < WARP_SIZE; i++)
    {
      if (thread_state[i].status != THREAD_EXITTED)
      {
        thread_state[i].status = THREAD_ENABLED;
        thread_state[i].next = NULL;
      }
    }
    return result;
  }
  // If we didn't converge, then we are going to go to next
  // Recompute the enabled and disabled threads
  for (int i = 0; i < WARP_SIZE; i++)
  {
    // Skip all the exitted threads
    if (targets[i] == NULL)
      continue;
    // Enable any threads going to next
    if (targets[i] == next)
    {
      thread_state[i].status = THREAD_ENABLED;
      thread_state[i].next = NULL;
    }
    else if (thread_state[i].status == THREAD_ENABLED)
    {
      // Disable all threads not going to next that 
      // weren't already disabled to begin with
      assert(targets[i] == target);
      thread_state[i].status = THREAD_DISABLED;
      thread_state[i].next = target;
    }
  }
  return next;
}

void PTXBranch::set_targets(const std::map<std::string,PTXLabel*> &labels)
{
  std::map<std::string,PTXLabel*>::const_iterator finder =
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
    int64_t address = shared_offset * SDDRINC;
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
               bool imm, bool pred, int line_num)
  : PTXInstruction(PTX_AND, line_num), 
    immediate(imm), predicate(pred)
{
  args[0] = zero;
  args[1] = one;
  args[2] = two;
}

PTXInstruction* PTXAnd::emulate(Thread *thread)
{
  if (predicate)
  {
    if (immediate)
    {
      bool source;
      if (!thread->get_pred(args[1], source))
        return next;
      bool value = source && bool(args[2]);
      thread->set_pred(args[0], value);
    }
    else
    {
      int64_t source, other; 
      if (!thread->get_value(args[1], source))
        return next;
      if (!thread->get_value(args[2], other))
        return next;
      bool value = source && other;
      thread->set_pred(args[0], value);
    }
  }
  else
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
    result = new PTXAnd(arg1, arg2, arg3, immediate, 
                        false/*pred*/, line_num);
    return true;
  }
  else if (line.find("and.pred") != std::string::npos)
  {
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    assert(tokens.size() == 4);
    bool negate;
    int64_t arg1 = parse_predicate(tokens[1], negate);
    int64_t arg2 = parse_predicate(tokens[2], negate);
    const size_t regs = count(line, "%");
    assert((regs == 2) || (regs == 3));
    const bool immediate = (regs == 2);
    int64_t arg3 = immediate ? parse_immediate(tokens[3])
                             : parse_predicate(tokens[3], negate);
    result = new PTXAnd(arg1, arg2, arg3, immediate, 
                        true/*pred*/, line_num);
    return true;
  }
  return false;
}

PTXOr::PTXOr(int64_t zero, int64_t one, int64_t two, 
             bool imm, bool pred, int line_num)
  : PTXInstruction(PTX_OR, line_num), 
    immediate(imm), predicate(pred)
{
  args[0] = zero;
  args[1] = one;
  args[2] = two;
}

PTXInstruction* PTXOr::emulate(Thread *thread)
{
  if (predicate)
  {
    if (immediate)
    {
      bool source;
      if (!thread->get_pred(args[1], source))
        return next;
      bool value = source || bool(args[2]);
      thread->set_pred(args[0], value);
    }
    else
    {
      bool source, other;
      if (!thread->get_pred(args[1], source))
        return next;
      if (!thread->get_pred(args[2], other))
        return next;
      bool value = source || other;
      thread->set_pred(args[0], value);
    }
  }
  else
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
    result = new PTXOr(arg1, arg2, arg3, immediate, 
                       false/*predi*/, line_num);
    return true;
  }
  else if (line.find("or.pred") != std::string::npos)
  {
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    assert(tokens.size() == 4);
    bool negate;
    int64_t arg1 = parse_predicate(tokens[1], negate);
    int64_t arg2 = parse_predicate(tokens[2], negate);
    const size_t regs = count(line, "%");
    assert((regs == 2) || (regs == 3));
    const bool immediate = (regs == 2);
    int64_t arg3 = immediate ? parse_immediate(tokens[3])
                             : parse_predicate(tokens[3], negate);
    result = new PTXOr(arg1, arg2, arg3, immediate, 
                       true/*predi*/, line_num);
    return true;
  }
  return false;
}

PTXXor::PTXXor(int64_t zero, int64_t one, int64_t two, 
             bool imm, bool pred, int line_num)
  : PTXInstruction(PTX_XOR, line_num), 
    immediate(imm), predicate(pred)
{
  args[0] = zero;
  args[1] = one;
  args[2] = two;
}

PTXInstruction* PTXXor::emulate(Thread *thread)
{
  if (predicate)
  {
    if (immediate)
    {
      bool source;
      if (!thread->get_pred(args[1], source))
        return next;
      bool value = (source && !(bool(args[2]))) ||
                   (!source && bool(args[2]));;
      thread->set_pred(args[0], value);
    }
    else
    {
      bool source, other;
      if (!thread->get_pred(args[1], source))
        return next;
      if (!thread->get_pred(args[2], other))
        return next;
      bool value = (source && !other) || (!source && other);
      thread->set_pred(args[0], value);
    }
  }
  else
  {
    if (immediate)
    {
      int64_t source;
      if (!thread->get_value(args[1], source))
        return next;
      int64_t value = source ^ args[2];
      thread->set_value(args[0], value);
    }
    else
    {
      int64_t source, other;
      if (!thread->get_value(args[1], source))
        return next;
      if (!thread->get_value(args[2], other))
        return next;
      int64_t value = source ^ other;
      thread->set_value(args[0], value);
    }
  }
  return next;
}

/*static*/
bool PTXXor::interpret(const std::string &line, int line_num,
                      PTXInstruction *&result)
{
  if (line.find("xor.b") != std::string::npos)
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
    result = new PTXXor(arg1, arg2, arg3, immediate, 
                       false/*predi*/, line_num);
    return true;
  }
  else if (line.find("xor.pred") != std::string::npos)
  {
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    assert(tokens.size() == 4);
    bool negate;
    int64_t arg1 = parse_predicate(tokens[1], negate);
    int64_t arg2 = parse_predicate(tokens[2], negate);
    const size_t regs = count(line, "%");
    assert((regs == 2) || (regs == 3));
    const bool immediate = (regs == 2);
    int64_t arg3 = immediate ? parse_immediate(tokens[3])
                             : parse_predicate(tokens[3], negate);
    result = new PTXXor(arg1, arg2, arg3, immediate, 
                       true/*predi*/, line_num);
    return true;
  }
  return false;
}

PTXNot::PTXNot(int64_t zero, int64_t one, bool pred, int line_num)
  : PTXInstruction(PTX_NOT, line_num), predicate(pred)
{
  args[0] = zero;
  args[1] = one;
}

PTXInstruction* PTXNot::emulate(Thread *thread)
{
  if (predicate)
  {
    bool source;
    if (!thread->get_pred(args[1], source))
      return next;
    bool value = !source;
    thread->set_pred(args[0], value);
  }
  else
  {
    int64_t source;
    if (!thread->get_value(args[1], source))
      return next;
    int64_t value = !source;
    thread->set_value(args[0], value);
  }
  return next;
}

/*static*/
bool PTXNot::interpret(const std::string &line, int line_num,
                        PTXInstruction *&result)
{
  if (line.find("not.b") != std::string::npos)
  {
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    assert(tokens.size() == 3);
    int64_t arg1 = parse_register(tokens[1]);
    int64_t arg2 = parse_register(tokens[2]);
    result = new PTXNot(arg1, arg2, false/*pred*/, line_num);
    return true;
  }
  else if (line.find("not.pred") != std::string::npos)
  {
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    assert(tokens.size() == 3);
    bool negate;
    int64_t arg1 = parse_predicate(tokens[1], negate);
    int64_t arg2 = parse_predicate(tokens[2], negate);
    result = new PTXNot(arg1, arg2, true/*pred*/, line_num);
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
  WeftInstruction *instruction;
  if (sync)
    instruction = new BarrierSync(name_value, count_value, this, thread);
  else
    instruction = new BarrierArrive(name_value, count_value, this, thread);
  thread->add_instruction(instruction);
  thread->update_max_barrier_name(name_value);
  return next;
}

PTXInstruction* PTXBarrier::emulate_warp(Thread **threads,
                                         ThreadState *thread_state,
                                         int &shared_access_id,
                                         SharedStore &store)
{
  // In warp-synchronous execution, if any thread in a warp arrives
  // at a barrier, then it is like all of the threads in a warp arrived
  // regardless of whether the thread is enabled, disabled, or exitted 
  bool one_enabled = false;
  for (int i = 0; i < WARP_SIZE; i++)
  {
    if (thread_state[i].status == THREAD_ENABLED)
    {
      one_enabled = true;
      break;
    }
  }
  if (one_enabled)
  {
    for (int i = 0; i < WARP_SIZE; i++)
      emulate(threads[i]);
  }
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

PTXSharedAccess::PTXSharedAccess(int64_t ad, int64_t o, bool w, 
                                 bool has, int64_t ag, bool imm, int line_num)
  : PTXInstruction(PTX_SHARED_ACCESS, line_num), has_name(false),
    addr(ad), offset(o), arg(ag), write(w), has_arg(has), immediate(imm)
{
}

PTXSharedAccess::PTXSharedAccess(const std::string &n, int64_t o, bool w,
                                 bool has, int64_t a, bool imm, int line_num)
  : PTXInstruction(PTX_SHARED_ACCESS, line_num), has_name(true),
    name(n), offset(o), arg(a), write(w), has_arg(has), immediate(imm)
{
}

PTXInstruction* PTXSharedAccess::emulate(Thread *thread)
{
  int64_t value;
  if (has_name)
  {
    if (!thread->find_shared_location(name, value))
      return next;
  }
  else if (!thread->get_value(addr, value))
    return next;
  int64_t address = value + offset;
  WeftAccess *instruction;
  if (write)
    instruction = new SharedWrite(address, this, thread);
  else
    instruction = new SharedRead(address, this, thread);
  thread->add_instruction(instruction);
  thread->update_shared_memory(instruction);
  return next;
}

PTXInstruction* PTXSharedAccess::emulate_warp(Thread **threads,
                                              ThreadState *thread_state,
                                              int &shared_access_id,
                                              SharedStore &store)
{
  // Shared accesses work mostly the same, but we want to detect
  // races from threads in the same warp doing accesses at the 
  // same time to the same instruction, so we record a common identifier
  if (write)
  {
    for (int i = 0; i < WARP_SIZE; i++)
    {
      if (thread_state[i].status != THREAD_ENABLED)
        continue;
      int64_t addr_value;
      if (has_name)
      {
        if (!threads[i]->find_shared_location(name, addr_value))
          continue;
      }
      else if (!threads[i]->get_value(addr, addr_value))
        continue;
      int64_t address = addr_value + offset;
      WeftAccess *instruction = 
        new SharedWrite(address, this, threads[i], shared_access_id);
      threads[i]->add_instruction(instruction);
      threads[i]->update_shared_memory(instruction);
      if (has_arg)
      {
        if (!immediate)
        {
          int64_t value;
          if (threads[i]->get_value(arg, value))
            store.write(address, value);
        }
        else
          store.write(address, arg);
      }
    }
  }
  else
  {
    for (int i = 0; i < WARP_SIZE; i++)
    {
      if (thread_state[i].status != THREAD_ENABLED)
        continue;
      int64_t addr_value;
      if (has_name)
      {
        if (!threads[i]->find_shared_location(name, addr_value))
          continue;
      }
      else if (!threads[i]->get_value(addr, addr_value))
        continue;
      int64_t address = addr_value + offset;
      WeftAccess *instruction = 
        new SharedRead(address, this, threads[i], shared_access_id);
      threads[i]->add_instruction(instruction);
      threads[i]->update_shared_memory(instruction);
      if (has_arg)
      {
        assert(!immediate);
        int64_t value;
        if (store.read(address, value))
          threads[i]->set_value(arg, value);
      }
    }
  }
  // Increment the shared_access_id
  shared_access_id++;
  return next;
}

/*static*/
bool PTXSharedAccess::interpret(const std::string &line, int line_num,
                                PTXInstruction *&result)
{
  if ((line.find(".shared.") != std::string::npos) &&
      (line.find(".align.") == std::string::npos))
  {
    
    bool write = (line.find("st.") != std::string::npos);
    int64_t addr = 0;
    int64_t offset = 0;   
    std::string name;
    bool has_name = false;
    // First check to see if it has an offset
    if (line.find("+") != std::string::npos)
    {
      // Offset
      int start = line.find("[") + 1;
      int end = line.find("+");
      name = line.substr(start, end - start);
      if (name.find("%") != std::string::npos)
        addr = parse_register(name);
      else
        has_name = true;
      // Now parse the offset
      offset = parse_immediate(line.substr(end+1));
    }
    else
    {
      // No Offset
      int start = line.find("[") + 1;
      int end = line.find("]");
      name = line.substr(start, end - start);
      if (name.find("%") != std::string::npos)
        addr = parse_register(name);
      else
        has_name = true;
    }
    // Now parse the other argument
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    bool has_arg = false;
    bool immediate = false;
    int64_t arg = 0;
    if (tokens.size() == 3)
    {
      has_arg = true;
      if (write)
      {
        immediate = (tokens[2].find("%") == std::string::npos);
        if (immediate)
          arg = parse_immediate(tokens[2]);
        else
          arg = parse_register(tokens[2]);
      }
      else
      {
        immediate = (tokens[1].find("%") == std::string::npos);
        if (immediate)
          arg = parse_immediate(tokens[1]);
        else
          arg = parse_register(tokens[1]);
      }
    }
    if (has_name)
      result = new PTXSharedAccess(name, offset, write, has_arg,
                                   arg, immediate, line_num);
    else
      result = new PTXSharedAccess(addr, offset, write, has_arg, 
                                   arg, immediate, line_num);
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
  : PTXInstruction(PTX_CONVERT_ADDRESS, line_num), 
    has_name(false), src(one), dst(zero)
{
}

PTXConvertAddress::PTXConvertAddress(int64_t zero, const std::string &n, int line_num)
  : PTXInstruction(PTX_CONVERT_ADDRESS, line_num),
    has_name(true), dst(zero), name(n)
{
}

PTXInstruction* PTXConvertAddress::emulate(Thread *thread)
{
  if (!has_name)
  {
    int64_t value;
    if (!thread->get_value(src, value))
      return next;
    thread->set_value(dst, value);
  }
  else
  {
    int64_t value;
    if (thread->get_global_location(name.c_str(), value))
      thread->set_value(dst, value);
  }
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
    if (tokens[2].find("%") != std::string::npos)
    {
      int64_t arg2 = parse_register(tokens[2]);
      result = new PTXConvertAddress(arg1, arg2, line_num);
    }
    else
    {
      std::string name = tokens[2].substr(0, tokens[2].size()-1);
      result = new PTXConvertAddress(arg1, name, line_num);
    }
    return true;
  }
  return false;
}

PTXBitFieldExtract::PTXBitFieldExtract(int64_t a[4], bool imm[4], int line_num)
  : PTXInstruction(PTX_BFE, line_num)
{
  for (int i = 0; i < 4; i++)
    args[i] = a[i];
  for (int i = 0; i < 4; i++)
    immediate[i] = imm[i];
}

PTXInstruction* PTXBitFieldExtract::emulate(Thread *thread)
{
  int64_t vals[3];
  for (int i = 0; i < 3; i++)
  {
    if (immediate[i+1])
      vals[i] = args[i+1];
    else if (!thread->get_value(args[i+1], vals[i]))
      return next;
  }
  int index = vals[1] & 0xff;
  int length = vals[2] & 0xff;
  int64_t mask = 0;
  for (int i = index; i < (index+length); i++)
    mask |= (1 << i);
  int64_t value = (vals[0] & mask) >> index;
  thread->set_value(args[0], value);
  return next;
}

/*static*/
bool PTXBitFieldExtract::interpret(const std::string &line, int line_num,
                                   PTXInstruction *&result)
{
  if (line.find("bfe.") != std::string::npos)
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
    result = new PTXBitFieldExtract(args, immediate, line_num);
    return true;
  }
  return false;
}

PTXShuffle::PTXShuffle(ShuffleKind k, int64_t a[4], bool imm[4], int line_num)
  : PTXInstruction(PTX_SHFL, line_num), kind(k)
{
  for (int i = 0; i < 4; i++)
    args[i] = a[i];
  for (int i = 0; i < 4; i++)
    immediate[i] = imm[i];
}

PTXInstruction* PTXShuffle::emulate(Thread *thread)
{
  // This should never be called in single thread mode
  assert(false);
  return next;
}

PTXInstruction* PTXShuffle::emulate_warp(Thread **threads,
                                         ThreadState *thread_state,
                                         int &shared_access_id,
                                         SharedStore &store)
{
  // Compute the inputs from each thread
  int64_t inputs[WARP_SIZE];
  for (int i = 0; i < WARP_SIZE; i++)
  {
    if (thread_state[i].status != THREAD_ENABLED)
    {
      if (threads[i]->program->weft->report_warnings())
        fprintf(stderr,"WARNING: Shuffle with masked off thread!\n");
      inputs[i] = 0;
      continue;
    }
    if (immediate[1])
      inputs[i] = args[1];
    else
    {
      // Not an immediate, so evaluate it
      int64_t value;
      if (!threads[i]->get_value(args[1], value))
      {
        if (threads[i]->program->weft->report_warnings())
          fprintf(stderr,"WARNING: Shuffle depends on undefined value!\n");
        inputs[i] = 0;
        continue;
      }
      inputs[i] = value;
    }
  }
  // Now that we've got all the inputs, compute the shuffle
  for (int lane = 0; lane < WARP_SIZE; lane++)
  {
    int64_t b, c, dst;
    if (immediate[2])
      b = args[2];
    else if (!threads[lane]->get_value(args[2], b))
      continue;
    if (immediate[3])
      c = args[3];
    else if (!threads[lane]->get_value(args[3], c))
      continue;
    assert(!immediate[0]);
    if (!threads[lane]->get_value(args[0], dst))
      continue;

    int bval = b & 0x1f;
    int mask = (c & 0x1f00) >> 8;
    int cval = c & 0x1f;
    int minlane = lane & mask;
    int maxlane = minlane | (cval & ~mask);

    int src;
    bool valid;
    switch (kind)
    {
      case SHUFFLE_UP:
        {
          src = lane - bval;
          valid = (src >= maxlane);
          break;
        }
      case SHUFFLE_DOWN:
        {
          src = lane + bval;
          valid = (src <= maxlane);
          break;
        }
      case SHUFFLE_BUTTERFLY:
        {
          src = lane ^ bval;
          valid = (src <= maxlane);
          break;
        }
      case SHUFFLE_IDX:
        {
          src = minlane | (bval & ~mask);
          valid = (src <= maxlane);
          break;
        }
      default:
        assert(false); // should never get here
    }
    int64_t value;
    if (!valid)
      value = inputs[lane];
    else
      value = inputs[src];
    threads[lane]->set_value(dst, value);
  }
  return next;
}

/*static*/
bool PTXShuffle::interpret(const std::string &line, int line_num,
                           PTXInstruction *&result)
{
  if (line.find("shfl.") != std::string::npos)
  {
    ShuffleKind kind;
    if (line.find(".up") != std::string::npos)
      kind = SHUFFLE_UP;
    else if (line.find(".down") != std::string::npos)
      kind = SHUFFLE_DOWN;
    else if (line.find(".bfly") != std::string::npos)
      kind = SHUFFLE_BUTTERFLY;
    else
    {
      assert(line.find(".idx") != std::string::npos);
      kind = SHUFFLE_IDX;
    }
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
    result = new PTXShuffle(kind, args, immediate, line_num);
    return true;
  }
  return false;
}

PTXExit::PTXExit(int line_num)
  : PTXInstruction(PTX_EXIT, line_num), has_predicate(false)
{
}

PTXExit::PTXExit(int64_t pred, bool neg, int line_num)
  : PTXInstruction(PTX_EXIT, line_num), 
    has_predicate(true), negate(neg), predicate(pred)
{
}

PTXInstruction* PTXExit::emulate(Thread *thread)
{
  // If we're not predicated, then we are done
  if (!has_predicate)
    return NULL;
  // Otherwise evaluate the predicate
  bool value;
  if (!thread->get_pred(predicate, value))
  {
    if (thread->program->weft->report_warnings())
    {
      char buffer[11];
      decompress_identifier(predicate, buffer, 11);
      fprintf(stderr,"WEFT WARNING: Exit depends on undefined "
                     "predicate %s\n", buffer);
    }
    // Assume we don't exit
    return next;
  }
  if (negate)
    value = !value;
  if (value)
    return NULL;
  return next;
}

PTXInstruction* PTXExit::emulate_warp(Thread **threads,
                                      ThreadState *thread_state,
                                      int &shared_access_id,
                                      SharedStore &store)
{
  // Evaluate this for all the enabled threads
  for (int i = 0; i < WARP_SIZE; i++)
  {
    if (thread_state[i].status == THREAD_ENABLED)
    {
      if (!has_predicate)
      {
        thread_state[i].status = THREAD_EXITTED;
        continue;
      }
      bool value;
      if (!threads[i]->get_pred(predicate, value))
      {
        if (threads[i]->program->weft->report_warnings())
        {
          char buffer[11];
          decompress_identifier(predicate, buffer, 11);
          fprintf(stderr,"WEFT WARNING: Exit depends on undefined "
                         "predicate %s\n", buffer);
        }
        // Assume we don't exit
        continue;
      }
      if (negate)
        value = !value;
      if (value)
        thread_state[i].status = THREAD_EXITTED;
    }
  }
  // Check to see if we're all done
  bool converged = true;
  for (int i = 0; i < WARP_SIZE; i++)
  {
    if (thread_state[i].status != THREAD_EXITTED)
    {
      converged = false;
      break;
    }
  }
  if (converged)
    return NULL;
  return next;
}

/*static*/
bool PTXExit::interpret(const std::string &line, int line_num,
                        PTXInstruction *&result)
{
  // We'll model both return and exit the same
  if ((line.find("exit") != std::string::npos) ||
      (line.find("ret") != std::string::npos))
  {
    std::vector<std::string> tokens; 
    split(tokens, line.c_str());
    if (tokens.size() == 2)
    {
      bool negate;
      int64_t predicate = parse_predicate(tokens[0], negate);
      result = new PTXExit(predicate, negate, line_num);
    }
    else if(tokens.size() == 1)
      result = new PTXExit(line_num);
    else
      assert(false);
    return true;
  }
  return false;
}

PTXGlobalDecl::PTXGlobalDecl(char *n, int *v, size_t s, int line_num)
  : PTXInstruction(PTX_GLOBAL_DECL, line_num), name(n), values(v), size(s)
{
}

PTXInstruction* PTXGlobalDecl::emulate(Thread *thread)
{
  // Register the name of the memory and it's values with the thread
  thread->register_global_location(name, values, size);
  return next;
}

/*static*/
bool PTXGlobalDecl::interpret(const std::string &line, int line_num,
                              PTXInstruction *&result)
{
  if ((line.find(".const .align") != std::string::npos) ||
      (line.find(".global .align") != std::string::npos))
  {
    // Assume byte loading
    int start = line.find(".b8");
    // Jump to the start of the name
    int name_start = start + 4;
    int name_end = line.find("[");
    std::string name = line.substr(name_start, name_end - name_start);
    int size_start = name_end+1;
    int size_end = line.find("]");
    std::string size_str = line.substr(size_start, size_start - size_end);
    size_t bytes = atoi(size_str.c_str());
    assert((bytes % 4) == 0);
    size_t size = bytes/4;
    int *values = (int*)malloc(bytes);
    // Read in the numbers
    int index = line.find("{");
    for (unsigned i = 0; i < size; i++)
    {
      int value = 0;
      for (int j = 0; j < 4; j++)
      {
        // Read until we get to a number
        while ((line[index] < '0') ||
               (line[index] > '9'))
          index++;
        // We know these are never longer than 4 bytes;
        char buffer[4];
        int local_index = 0;
        while ((line[index] >= '0') &&
               (line[index] <= '9'))
          buffer[local_index++] = line[index++];
        assert(local_index < 4);
        buffer[local_index] = '\0';
        int temp = atoi(buffer);
        value |= (temp << (j*8));
      }
      values[i] = value;
    }
    result = new PTXGlobalDecl(strdup(name.c_str()), values, size, line_num);
    return true;
  }
  return false;
}

PTXGlobalLoad::PTXGlobalLoad(int64_t d, int64_t a, int line_num)
  : PTXInstruction(PTX_GLOBAL_LOAD, line_num), dst(d), addr(a)
{
}

PTXInstruction* PTXGlobalLoad::emulate(Thread *thread)
{
  int64_t value;
  if (thread->get_global_value(addr, value))
    thread->set_value(dst, value);
  return next;
}

/*static*/
bool PTXGlobalLoad::interpret(const std::string &line, int line_num,
                              PTXInstruction *&result)
{
  if (line.find("ld.global") != std::string::npos)
  {
    std::vector<std::string> tokens;
    split(tokens, line.c_str());
    assert(tokens.size() == 3);
    int arg1 = parse_register(tokens[1]);
    int arg2 = parse_register(tokens[2]);
    result = new PTXGlobalLoad(arg1, arg2, line_num);
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

void BarrierSync::print_instruction(FILE *target)
{
  fprintf(target,"bar.sync %d, %d;\n", name, count);
}

BarrierArrive::BarrierArrive(int n, int c, PTXBarrier *bar, Thread *thread)
  : WeftBarrier(n, c, bar, thread)
{
}

void BarrierArrive::print_instruction(FILE *target)
{
  fprintf(target,"bar.arrive %d, %d;\n", name, count);
}

WeftAccess::WeftAccess(int addr, PTXSharedAccess *acc, 
                       Thread *thread, int acc_id)
  : WeftInstruction(acc, thread), address(addr), access(acc), access_id(acc_id)
{
}

bool WeftAccess::has_happens_relationship(WeftAccess *other)
{
  // If they are the same thread, then we are done
  if (thread == other->thread)
    return true;
  assert(happens_relationship != NULL);
  int thread_id = other->thread->thread_id;
  int line_number = other->thread_line_number;
  return happens_relationship->has_happens(thread_id, line_number);
}

bool WeftAccess::is_warp_synchronous(WeftAccess *other)
{
  // Check to see if the threads are in the same warp
  int local_wid = thread->thread_id/WARP_SIZE;
  int other_wid = other->thread->thread_id/WARP_SIZE;
  if (local_wid != other_wid)
    return false;
  // We should only be here if we assumed warp synchronous
  // execution and therefore access IDs are non-negative.
  assert(access_id >= 0);
  assert(other->access_id >= 0);
  // If they are in the same warp, check to see if they have
  // the same access_id.  If they do, then we cannot use 
  // warp-synchronous execution to avoid the race test.
  // If they come from different access IDs then we know
  // they can't have happened at the same time because
  // warps execute in lock step.
  return (access_id != other->access_id);
}

SharedWrite::SharedWrite(int addr, PTXSharedAccess *acc, 
                         Thread *thread, int acc_id /*=-1*/)
  : WeftAccess(addr, acc, thread, acc_id)
{
}

void SharedWrite::print_instruction(FILE *target)
{
  fprintf(target,"write shared[%d];\n", address);
}

SharedRead::SharedRead(int addr, PTXSharedAccess *acc, 
                       Thread *thread, int acc_id /*=-1*/)
  : WeftAccess(addr, acc, thread, acc_id)
{
}

void SharedRead::print_instruction(FILE *target)
{
  fprintf(target,"read shared[%d];\n", address);
}

