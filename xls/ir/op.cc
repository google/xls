// Copyright 2024 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xls/ir/op.h"

#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace xls {

Op FromOpProto(OpProto op_proto) {
  switch (op_proto) {
    case OP_INVALID:
      LOG(FATAL) << "OP_INVALID received";
      break;
    case OP_ADD:
      return Op::kAdd;
    case OP_AFTER_ALL:
      return Op::kAfterAll;
    case OP_AND:
      return Op::kAnd;
    case OP_AND_REDUCE:
      return Op::kAndReduce;
    case OP_ARRAY:
      return Op::kArray;
    case OP_ARRAY_CONCAT:
      return Op::kArrayConcat;
    case OP_ARRAY_INDEX:
      return Op::kArrayIndex;
    case OP_ARRAY_SLICE:
      return Op::kArraySlice;
    case OP_ARRAY_UPDATE:
      return Op::kArrayUpdate;
    case OP_ASSERT:
      return Op::kAssert;
    case OP_BIT_SLICE:
      return Op::kBitSlice;
    case OP_BIT_SLICE_UPDATE:
      return Op::kBitSliceUpdate;
    case OP_CONCAT:
      return Op::kConcat;
    case OP_COUNTED_FOR:
      return Op::kCountedFor;
    case OP_COVER:
      return Op::kCover;
    case OP_DECODE:
      return Op::kDecode;
    case OP_DYNAMIC_BIT_SLICE:
      return Op::kDynamicBitSlice;
    case OP_DYNAMIC_COUNTED_FOR:
      return Op::kDynamicCountedFor;
    case OP_ENCODE:
      return Op::kEncode;
    case OP_EQ:
      return Op::kEq;
    case OP_GATE:
      return Op::kGate;
    case OP_IDENTITY:
      return Op::kIdentity;
    case OP_INPUT_PORT:
      return Op::kInputPort;
    case OP_INSTANTIATION_INPUT:
      return Op::kInstantiationInput;
    case OP_INSTANTIATION_OUTPUT:
      return Op::kInstantiationOutput;
    case OP_INVOKE:
      return Op::kInvoke;
    case OP_LITERAL:
      return Op::kLiteral;
    case OP_MAP:
      return Op::kMap;
    case OP_MIN_DELAY:
      return Op::kMinDelay;
    case OP_NAND:
      return Op::kNand;
    case OP_NE:
      return Op::kNe;
    case OP_NEG:
      return Op::kNeg;
    case OP_NEXT_VALUE:
      return Op::kNext;
    case OP_NOR:
      return Op::kNor;
    case OP_NOT:
      return Op::kNot;
    case OP_ONE_HOT:
      return Op::kOneHot;
    case OP_ONE_HOT_SEL:
      return Op::kOneHotSel;
    case OP_OR:
      return Op::kOr;
    case OP_OR_REDUCE:
      return Op::kOrReduce;
    case OP_OUTPUT_PORT:
      return Op::kOutputPort;
    case OP_PARAM:
      return Op::kParam;
    case OP_PRIORITY_SEL:
      return Op::kPrioritySel;
    case OP_RECEIVE:
      return Op::kReceive;
    case OP_REGISTER_READ:
      return Op::kRegisterRead;
    case OP_REGISTER_WRITE:
      return Op::kRegisterWrite;
    case OP_REVERSE:
      return Op::kReverse;
    case OP_SDIV:
      return Op::kSDiv;
    case OP_SEL:
      return Op::kSel;
    case OP_SEND:
      return Op::kSend;
    case OP_SGE:
      return Op::kSGe;
    case OP_SGT:
      return Op::kSGt;
    case OP_SHLL:
      return Op::kShll;
    case OP_SHRA:
      return Op::kShra;
    case OP_SHRL:
      return Op::kShrl;
    case OP_SIGN_EXT:
      return Op::kSignExt;
    case OP_SLE:
      return Op::kSLe;
    case OP_SLT:
      return Op::kSLt;
    case OP_SMOD:
      return Op::kSMod;
    case OP_SMUL:
      return Op::kSMul;
    case OP_SMULP:
      return Op::kSMulp;
    case OP_SUB:
      return Op::kSub;
    case OP_TRACE:
      return Op::kTrace;
    case OP_TUPLE:
      return Op::kTuple;
    case OP_TUPLE_INDEX:
      return Op::kTupleIndex;
    case OP_UDIV:
      return Op::kUDiv;
    case OP_UGE:
      return Op::kUGe;
    case OP_UGT:
      return Op::kUGt;
    case OP_ULE:
      return Op::kULe;
    case OP_ULT:
      return Op::kULt;
    case OP_UMOD:
      return Op::kUMod;
    case OP_UMUL:
      return Op::kUMul;
    case OP_UMULP:
      return Op::kUMulp;
    case OP_XOR:
      return Op::kXor;
    case OP_XOR_REDUCE:
      return Op::kXorReduce;
    case OP_ZERO_EXT:
      return Op::kZeroExt;
      // Note: since this is a proto enum there are sentinel values defined in
      // addition to the "real" ones above, which is why the enumeration above
      // is not exhaustive.
    default:
      LOG(FATAL) << "Invalid OpProto: " << static_cast<int64_t>(op_proto);
  }
}

OpProto ToOpProto(Op op) {
  switch (op) {
    case Op::kAdd:
      return OP_ADD;
    case Op::kAnd:
      return OP_AND;
    case Op::kAndReduce:
      return OP_AND_REDUCE;
    case Op::kAssert:
      return OP_ASSERT;
    case Op::kCover:
      return OP_COVER;
    case Op::kReceive:
      return OP_RECEIVE;
    case Op::kSend:
      return OP_SEND;
    case Op::kNand:
      return OP_NAND;
    case Op::kNor:
      return OP_NOR;
    case Op::kAfterAll:
      return OP_AFTER_ALL;
    case Op::kArray:
      return OP_ARRAY;
    case Op::kArrayIndex:
      return OP_ARRAY_INDEX;
    case Op::kArraySlice:
      return OP_ARRAY_SLICE;
    case Op::kArrayUpdate:
      return OP_ARRAY_UPDATE;
    case Op::kArrayConcat:
      return OP_ARRAY_CONCAT;
    case Op::kBitSlice:
      return OP_BIT_SLICE;
    case Op::kDynamicBitSlice:
      return OP_DYNAMIC_BIT_SLICE;
    case Op::kBitSliceUpdate:
      return OP_BIT_SLICE_UPDATE;
    case Op::kConcat:
      return OP_CONCAT;
    case Op::kCountedFor:
      return OP_COUNTED_FOR;
    case Op::kDecode:
      return OP_DECODE;
    case Op::kDynamicCountedFor:
      return OP_DYNAMIC_COUNTED_FOR;
    case Op::kEncode:
      return OP_ENCODE;
    case Op::kEq:
      return OP_EQ;
    case Op::kIdentity:
      return OP_IDENTITY;
    case Op::kInvoke:
      return OP_INVOKE;
    case Op::kInputPort:
      return OP_INPUT_PORT;
    case Op::kLiteral:
      return OP_LITERAL;
    case Op::kMap:
      return OP_MAP;
    case Op::kNe:
      return OP_NE;
    case Op::kNeg:
      return OP_NEG;
    case Op::kNot:
      return OP_NOT;
    case Op::kOneHot:
      return OP_ONE_HOT;
    case Op::kOneHotSel:
      return OP_ONE_HOT_SEL;
    case Op::kPrioritySel:
      return OP_PRIORITY_SEL;
    case Op::kOr:
      return OP_OR;
    case Op::kOrReduce:
      return OP_OR_REDUCE;
    case Op::kOutputPort:
      return OP_OUTPUT_PORT;
    case Op::kParam:
      return OP_PARAM;
    case Op::kNext:
      return OP_NEXT_VALUE;
    case Op::kRegisterRead:
      return OP_REGISTER_READ;
    case Op::kRegisterWrite:
      return OP_REGISTER_WRITE;
    case Op::kInstantiationOutput:
      return OP_INSTANTIATION_OUTPUT;
    case Op::kInstantiationInput:
      return OP_INSTANTIATION_INPUT;
    case Op::kReverse:
      return OP_REVERSE;
    case Op::kSDiv:
      return OP_SDIV;
    case Op::kSMod:
      return OP_SMOD;
    case Op::kSel:
      return OP_SEL;
    case Op::kSGe:
      return OP_SGE;
    case Op::kSGt:
      return OP_SGT;
    case Op::kShll:
      return OP_SHLL;
    case Op::kShrl:
      return OP_SHRL;
    case Op::kShra:
      return OP_SHRA;
    case Op::kSignExt:
      return OP_SIGN_EXT;
    case Op::kSLe:
      return OP_SLE;
    case Op::kSLt:
      return OP_SLT;
    case Op::kSMul:
      return OP_SMUL;
    case Op::kSMulp:
      return OP_SMULP;
    case Op::kSub:
      return OP_SUB;
    case Op::kTuple:
      return OP_TUPLE;
    case Op::kTupleIndex:
      return OP_TUPLE_INDEX;
    case Op::kUDiv:
      return OP_UDIV;
    case Op::kUMod:
      return OP_UMOD;
    case Op::kUGe:
      return OP_UGE;
    case Op::kUGt:
      return OP_UGT;
    case Op::kULe:
      return OP_ULE;
    case Op::kULt:
      return OP_ULT;
    case Op::kUMul:
      return OP_UMUL;
    case Op::kUMulp:
      return OP_UMULP;
    case Op::kXor:
      return OP_XOR;
    case Op::kXorReduce:
      return OP_XOR_REDUCE;
    case Op::kZeroExt:
      return OP_ZERO_EXT;
    case Op::kGate:
      return OP_GATE;
    case Op::kTrace:
      return OP_TRACE;
    case Op::kMinDelay:
      return OP_MIN_DELAY;
  }
  LOG(FATAL) << "Invalid Op: " << static_cast<int64_t>(op);
}

std::string OpToString(Op op) {
  static const absl::NoDestructor<absl::flat_hash_map<Op, std::string>> op_map({
      {Op::kAdd, "add"},
      {Op::kAfterAll, "after_all"},
      {Op::kAnd, "and"},
      {Op::kAndReduce, "and_reduce"},
      {Op::kArray, "array"},
      {Op::kArrayConcat, "array_concat"},
      {Op::kArrayIndex, "array_index"},
      {Op::kArraySlice, "array_slice"},
      {Op::kArrayUpdate, "array_update"},
      {Op::kAssert, "assert"},
      {Op::kBitSlice, "bit_slice"},
      {Op::kBitSliceUpdate, "bit_slice_update"},
      {Op::kConcat, "concat"},
      {Op::kCountedFor, "counted_for"},
      {Op::kCover, "cover"},
      {Op::kDecode, "decode"},
      {Op::kDynamicBitSlice, "dynamic_bit_slice"},
      {Op::kDynamicCountedFor, "dynamic_counted_for"},
      {Op::kEncode, "encode"},
      {Op::kEq, "eq"},
      {Op::kGate, "gate"},
      {Op::kIdentity, "identity"},
      {Op::kInputPort, "input_port"},
      {Op::kInstantiationInput, "instantiation_input"},
      {Op::kInstantiationOutput, "instantiation_output"},
      {Op::kInvoke, "invoke"},
      {Op::kLiteral, "literal"},
      {Op::kMap, "map"},
      {Op::kMinDelay, "min_delay"},
      {Op::kNand, "nand"},
      {Op::kNe, "ne"},
      {Op::kNeg, "neg"},
      {Op::kNext, "next_value"},
      {Op::kNor, "nor"},
      {Op::kNot, "not"},
      {Op::kOneHot, "one_hot"},
      {Op::kOneHotSel, "one_hot_sel"},
      {Op::kOr, "or"},
      {Op::kOrReduce, "or_reduce"},
      {Op::kOutputPort, "output_port"},
      {Op::kParam, "param"},
      {Op::kPrioritySel, "priority_sel"},
      {Op::kReceive, "receive"},
      {Op::kRegisterRead, "register_read"},
      {Op::kRegisterWrite, "register_write"},
      {Op::kReverse, "reverse"},
      {Op::kSDiv, "sdiv"},
      {Op::kSGe, "sge"},
      {Op::kSGt, "sgt"},
      {Op::kSLe, "sle"},
      {Op::kSLt, "slt"},
      {Op::kSMod, "smod"},
      {Op::kSMul, "smul"},
      {Op::kSMulp, "smulp"},
      {Op::kSel, "sel"},
      {Op::kSend, "send"},
      {Op::kShll, "shll"},
      {Op::kShra, "shra"},
      {Op::kShrl, "shrl"},
      {Op::kSignExt, "sign_ext"},
      {Op::kSub, "sub"},
      {Op::kTrace, "trace"},
      {Op::kTuple, "tuple"},
      {Op::kTupleIndex, "tuple_index"},
      {Op::kUDiv, "udiv"},
      {Op::kUGe, "uge"},
      {Op::kUGt, "ugt"},
      {Op::kULe, "ule"},
      {Op::kULt, "ult"},
      {Op::kUMod, "umod"},
      {Op::kUMul, "umul"},
      {Op::kUMulp, "umulp"},
      {Op::kXor, "xor"},
      {Op::kXorReduce, "xor_reduce"},
      {Op::kZeroExt, "zero_ext"},
  });
  auto found = op_map->find(op);
  if (found == op_map->end()) {
    LOG(FATAL) << "OpToString(" << static_cast<uint64_t>(op)
               << ") failed, unknown op";
  }
  return found->second;
}

absl::StatusOr<Op> StringToOp(std::string_view op_str) {
  static const absl::NoDestructor<absl::flat_hash_map<std::string, Op>>
      string_map({
          {"add", Op::kAdd},
          {"after_all", Op::kAfterAll},
          {"and", Op::kAnd},
          {"and_reduce", Op::kAndReduce},
          {"array", Op::kArray},
          {"array_concat", Op::kArrayConcat},
          {"array_index", Op::kArrayIndex},
          {"array_slice", Op::kArraySlice},
          {"array_update", Op::kArrayUpdate},
          {"assert", Op::kAssert},
          {"bit_slice", Op::kBitSlice},
          {"bit_slice_update", Op::kBitSliceUpdate},
          {"concat", Op::kConcat},
          {"counted_for", Op::kCountedFor},
          {"cover", Op::kCover},
          {"decode", Op::kDecode},
          {"dynamic_bit_slice", Op::kDynamicBitSlice},
          {"dynamic_counted_for", Op::kDynamicCountedFor},
          {"encode", Op::kEncode},
          {"eq", Op::kEq},
          {"gate", Op::kGate},
          {"identity", Op::kIdentity},
          {"input_port", Op::kInputPort},
          {"instantiation_input", Op::kInstantiationInput},
          {"instantiation_output", Op::kInstantiationOutput},
          {"invoke", Op::kInvoke},
          {"literal", Op::kLiteral},
          {"map", Op::kMap},
          {"min_delay", Op::kMinDelay},
          {"nand", Op::kNand},
          {"ne", Op::kNe},
          {"neg", Op::kNeg},
          {"next_value", Op::kNext},
          {"nor", Op::kNor},
          {"not", Op::kNot},
          {"one_hot", Op::kOneHot},
          {"one_hot_sel", Op::kOneHotSel},
          {"or", Op::kOr},
          {"or_reduce", Op::kOrReduce},
          {"output_port", Op::kOutputPort},
          {"param", Op::kParam},
          {"priority_sel", Op::kPrioritySel},
          {"receive", Op::kReceive},
          {"register_read", Op::kRegisterRead},
          {"register_write", Op::kRegisterWrite},
          {"reverse", Op::kReverse},
          {"sdiv", Op::kSDiv},
          {"sel", Op::kSel},
          {"send", Op::kSend},
          {"sge", Op::kSGe},
          {"sgt", Op::kSGt},
          {"shll", Op::kShll},
          {"shra", Op::kShra},
          {"shrl", Op::kShrl},
          {"sign_ext", Op::kSignExt},
          {"sle", Op::kSLe},
          {"slt", Op::kSLt},
          {"smod", Op::kSMod},
          {"smul", Op::kSMul},
          {"smulp", Op::kSMulp},
          {"sub", Op::kSub},
          {"trace", Op::kTrace},
          {"tuple", Op::kTuple},
          {"tuple_index", Op::kTupleIndex},
          {"udiv", Op::kUDiv},
          {"uge", Op::kUGe},
          {"ugt", Op::kUGt},
          {"ule", Op::kULe},
          {"ult", Op::kULt},
          {"umod", Op::kUMod},
          {"umul", Op::kUMul},
          {"umulp", Op::kUMulp},
          {"xor", Op::kXor},
          {"xor_reduce", Op::kXorReduce},
          {"zero_ext", Op::kZeroExt},
      });
  auto found = string_map->find(op_str);
  if (found == string_map->end()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unknown operation for string-to-op conversion: ", op_str));
  }
  return found->second;
}

bool OpIsCompare(Op op) {
  if (op == Op::kEq) {
    return true;
  }
  if (op == Op::kNe) {
    return true;
  }
  if (op == Op::kSGe) {
    return true;
  }
  if (op == Op::kSGt) {
    return true;
  }
  if (op == Op::kSLe) {
    return true;
  }
  if (op == Op::kSLt) {
    return true;
  }
  if (op == Op::kUGe) {
    return true;
  }
  if (op == Op::kUGt) {
    return true;
  }
  if (op == Op::kULe) {
    return true;
  }
  if (op == Op::kULt) {
    return true;
  }
  return false;
}

bool OpIsAssociative(Op op) {
  if (op == Op::kAdd) {
    return true;
  }
  if (op == Op::kAnd) {
    return true;
  }
  if (op == Op::kAfterAll) {
    return true;
  }
  if (op == Op::kOr) {
    return true;
  }
  if (op == Op::kSMul) {
    return true;
  }
  if (op == Op::kSMulp) {
    return true;
  }
  if (op == Op::kUMul) {
    return true;
  }
  if (op == Op::kUMulp) {
    return true;
  }
  if (op == Op::kXor) {
    return true;
  }
  return false;
}

bool OpIsCommutative(Op op) {
  if (op == Op::kAdd) {
    return true;
  }
  if (op == Op::kAnd) {
    return true;
  }
  if (op == Op::kNand) {
    return true;
  }
  if (op == Op::kNor) {
    return true;
  }
  if (op == Op::kAfterAll) {
    return true;
  }
  if (op == Op::kEq) {
    return true;
  }
  if (op == Op::kNe) {
    return true;
  }
  if (op == Op::kOr) {
    return true;
  }
  if (op == Op::kSMul) {
    return true;
  }
  if (op == Op::kSMulp) {
    return true;
  }
  if (op == Op::kUMul) {
    return true;
  }
  if (op == Op::kUMulp) {
    return true;
  }
  if (op == Op::kXor) {
    return true;
  }
  return false;
}

bool OpIsBitWise(Op op) {
  if (op == Op::kAnd) {
    return true;
  }
  if (op == Op::kNand) {
    return true;
  }
  if (op == Op::kNor) {
    return true;
  }
  if (op == Op::kNot) {
    return true;
  }
  if (op == Op::kOr) {
    return true;
  }
  if (op == Op::kXor) {
    return true;
  }
  return false;
}

bool OpIsSideEffecting(Op op) {
  if (op == Op::kAssert) {
    return true;
  }
  if (op == Op::kCover) {
    return true;
  }
  if (op == Op::kReceive) {
    return true;
  }
  if (op == Op::kSend) {
    return true;
  }
  if (op == Op::kInputPort) {
    return true;
  }
  if (op == Op::kOutputPort) {
    return true;
  }
  if (op == Op::kParam) {
    return true;
  }
  if (op == Op::kNext) {
    return true;
  }
  if (op == Op::kRegisterRead) {
    return true;
  }
  if (op == Op::kRegisterWrite) {
    return true;
  }
  if (op == Op::kInstantiationOutput) {
    return true;
  }
  if (op == Op::kInstantiationInput) {
    return true;
  }
  if (op == Op::kGate) {
    return true;
  }
  if (op == Op::kTrace) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<AfterAll>(Op op) {
  if (op == Op::kAfterAll) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<MinDelay>(Op op) {
  if (op == Op::kMinDelay) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<Array>(Op op) {
  if (op == Op::kArray) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<ArrayIndex>(Op op) {
  if (op == Op::kArrayIndex) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<ArraySlice>(Op op) {
  if (op == Op::kArraySlice) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<ArrayUpdate>(Op op) {
  if (op == Op::kArrayUpdate) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<ArrayConcat>(Op op) {
  if (op == Op::kArrayConcat) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<BinOp>(Op op) {
  if (op == Op::kAdd) {
    return true;
  }
  if (op == Op::kSDiv) {
    return true;
  }
  if (op == Op::kSMod) {
    return true;
  }
  if (op == Op::kShll) {
    return true;
  }
  if (op == Op::kShrl) {
    return true;
  }
  if (op == Op::kShra) {
    return true;
  }
  if (op == Op::kSub) {
    return true;
  }
  if (op == Op::kUDiv) {
    return true;
  }
  if (op == Op::kUMod) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<ArithOp>(Op op) {
  if (op == Op::kSMul) {
    return true;
  }
  if (op == Op::kUMul) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<PartialProductOp>(Op op) {
  if (op == Op::kSMulp) {
    return true;
  }
  if (op == Op::kUMulp) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<Assert>(Op op) {
  if (op == Op::kAssert) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<Trace>(Op op) {
  if (op == Op::kTrace) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<Cover>(Op op) {
  if (op == Op::kCover) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<BitwiseReductionOp>(Op op) {
  if (op == Op::kAndReduce) {
    return true;
  }
  if (op == Op::kOrReduce) {
    return true;
  }
  if (op == Op::kXorReduce) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<Receive>(Op op) {
  if (op == Op::kReceive) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<Send>(Op op) {
  if (op == Op::kSend) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<NaryOp>(Op op) {
  if (op == Op::kAnd) {
    return true;
  }
  if (op == Op::kNand) {
    return true;
  }
  if (op == Op::kNor) {
    return true;
  }
  if (op == Op::kOr) {
    return true;
  }
  if (op == Op::kXor) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<BitSlice>(Op op) {
  if (op == Op::kBitSlice) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<DynamicBitSlice>(Op op) {
  if (op == Op::kDynamicBitSlice) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<BitSliceUpdate>(Op op) {
  if (op == Op::kBitSliceUpdate) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<CompareOp>(Op op) {
  if (op == Op::kEq) {
    return true;
  }
  if (op == Op::kNe) {
    return true;
  }
  if (op == Op::kSGe) {
    return true;
  }
  if (op == Op::kSGt) {
    return true;
  }
  if (op == Op::kSLe) {
    return true;
  }
  if (op == Op::kSLt) {
    return true;
  }
  if (op == Op::kUGe) {
    return true;
  }
  if (op == Op::kUGt) {
    return true;
  }
  if (op == Op::kULe) {
    return true;
  }
  if (op == Op::kULt) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<Concat>(Op op) {
  if (op == Op::kConcat) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<CountedFor>(Op op) {
  if (op == Op::kCountedFor) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<DynamicCountedFor>(Op op) {
  if (op == Op::kDynamicCountedFor) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<ExtendOp>(Op op) {
  if (op == Op::kSignExt) {
    return true;
  }
  if (op == Op::kZeroExt) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<Invoke>(Op op) {
  if (op == Op::kInvoke) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<Literal>(Op op) {
  if (op == Op::kLiteral) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<Map>(Op op) {
  if (op == Op::kMap) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<OneHot>(Op op) {
  if (op == Op::kOneHot) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<OneHotSelect>(Op op) {
  if (op == Op::kOneHotSel) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<PrioritySelect>(Op op) {
  if (op == Op::kPrioritySel) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<Param>(Op op) {
  if (op == Op::kParam) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<Next>(Op op) {
  if (op == Op::kNext) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<Select>(Op op) {
  if (op == Op::kSel) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<Tuple>(Op op) {
  if (op == Op::kTuple) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<TupleIndex>(Op op) {
  if (op == Op::kTupleIndex) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<UnOp>(Op op) {
  if (op == Op::kIdentity) {
    return true;
  }
  if (op == Op::kNeg) {
    return true;
  }
  if (op == Op::kNot) {
    return true;
  }
  if (op == Op::kReverse) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<Decode>(Op op) {
  if (op == Op::kDecode) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<Encode>(Op op) {
  if (op == Op::kEncode) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<InputPort>(Op op) {
  if (op == Op::kInputPort) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<OutputPort>(Op op) {
  if (op == Op::kOutputPort) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<RegisterRead>(Op op) {
  if (op == Op::kRegisterRead) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<RegisterWrite>(Op op) {
  if (op == Op::kRegisterWrite) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<InstantiationOutput>(Op op) {
  if (op == Op::kInstantiationOutput) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<InstantiationInput>(Op op) {
  if (op == Op::kInstantiationInput) {
    return true;
  }
  return false;
}

template <>
bool IsOpClass<Gate>(Op op) {
  if (op == Op::kGate) {
    return true;
  }
  return false;
}

std::ostream& operator<<(std::ostream& os, Op op) {
  os << OpToString(op);
  return os;
}

}  // namespace xls
