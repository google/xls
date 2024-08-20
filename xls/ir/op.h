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

#ifndef XLS_IR_OP_H_
#define XLS_IR_OP_H_

#include <array>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/op.pb.h"

// TODO(meheff): Add comments to classes and methods.

namespace xls {

// Enumerates the operator for nodes in the IR.
enum class Op : int8_t {
  kAdd,
  kAfterAll,
  kAnd,
  kAndReduce,
  kArray,
  kArrayConcat,
  kArrayIndex,
  kArraySlice,
  kArrayUpdate,
  kAssert,
  kBitSlice,
  kBitSliceUpdate,
  kConcat,
  kCountedFor,
  kCover,
  kDecode,
  kDynamicBitSlice,
  kDynamicCountedFor,
  kEncode,
  kEq,
  kGate,
  kIdentity,
  kInputPort,
  kInstantiationInput,
  kInstantiationOutput,
  kInvoke,
  kLiteral,
  kMap,
  kMinDelay,
  kNand,
  kNe,
  kNeg,
  kNext,
  kNor,
  kNot,
  kOneHot,
  kOneHotSel,
  kOr,
  kOrReduce,
  kOutputPort,
  kParam,
  kPrioritySel,
  kReceive,
  kRegisterRead,
  kRegisterWrite,
  kReverse,
  kSDiv,
  kSGe,
  kSGt,
  kSLe,
  kSLt,
  kSMod,
  kSMul,
  kSMulp,
  kSel,
  kSend,
  kShll,
  kShra,
  kShrl,
  kSignExt,
  kSub,
  kTrace,
  kTuple,
  kTupleIndex,
  kUDiv,
  kUGe,
  kUGt,
  kULe,
  kULt,
  kUMod,
  kUMul,
  kUMulp,
  kXor,
  kXorReduce,
  kZeroExt,
  kLast = Op::kZeroExt
};

constexpr std::array<Op, static_cast<int>(Op::kLast) + 1> kAllOps = {
    Op::kAdd,
    Op::kAfterAll,
    Op::kAnd,
    Op::kAndReduce,
    Op::kArray,
    Op::kArrayConcat,
    Op::kArrayIndex,
    Op::kArraySlice,
    Op::kArrayUpdate,
    Op::kAssert,
    Op::kBitSlice,
    Op::kBitSliceUpdate,
    Op::kConcat,
    Op::kCountedFor,
    Op::kCover,
    Op::kDecode,
    Op::kDynamicBitSlice,
    Op::kDynamicCountedFor,
    Op::kEncode,
    Op::kEq,
    Op::kGate,
    Op::kIdentity,
    Op::kInputPort,
    Op::kInstantiationInput,
    Op::kInstantiationOutput,
    Op::kInvoke,
    Op::kLiteral,
    Op::kMap,
    Op::kMinDelay,
    Op::kNand,
    Op::kNe,
    Op::kNeg,
    Op::kNext,
    Op::kNor,
    Op::kNot,
    Op::kOneHot,
    Op::kOneHotSel,
    Op::kOr,
    Op::kOrReduce,
    Op::kOutputPort,
    Op::kParam,
    Op::kPrioritySel,
    Op::kReceive,
    Op::kRegisterRead,
    Op::kRegisterWrite,
    Op::kReverse,
    Op::kSDiv,
    Op::kSGe,
    Op::kSGt,
    Op::kSLe,
    Op::kSLt,
    Op::kSMod,
    Op::kSMul,
    Op::kSMulp,
    Op::kSel,
    Op::kSend,
    Op::kShll,
    Op::kShra,
    Op::kShrl,
    Op::kSignExt,
    Op::kSub,
    Op::kTrace,
    Op::kTuple,
    Op::kTupleIndex,
    Op::kUDiv,
    Op::kUGe,
    Op::kUGt,
    Op::kULe,
    Op::kULt,
    Op::kUMod,
    Op::kUMul,
    Op::kUMulp,
    Op::kXor,
    Op::kXorReduce,
    Op::kZeroExt,
};

constexpr int64_t kOpLimit = kAllOps.size();

// Converts an OpProto into an Op.
Op FromOpProto(OpProto op_proto);

// Converts an Op into an OpProto.
OpProto ToOpProto(Op op);

// Converts the "op" enumeration to a human readable string.
std::string OpToString(Op op);

// Converts a human readable op string into the "op" enumeration.
absl::StatusOr<Op> StringToOp(std::string_view op_str);

// Returns whether the operation is a compare operation.
bool OpIsCompare(Op op);

// Returns whether the operation is associative, eg., kAdd, or kOr.
bool OpIsAssociative(Op op);

// Returns whether the operation is commutative, eg., kAdd, or kEq.
bool OpIsCommutative(Op op);

// Returns whether the operation is a bitwise logical op, eg., kAnd or kOr.
bool OpIsBitWise(Op op);

// Returns whether the operation has side effects, eg., kAssert, kSend.
bool OpIsSideEffecting(Op op);

// Forward declare all Op classes (subclasses of Node).
class AfterAll;
class MinDelay;
class Array;
class ArrayIndex;
class ArraySlice;
class ArrayUpdate;
class ArrayConcat;
class BinOp;
class ArithOp;
class PartialProductOp;
class Assert;
class Trace;
class Cover;
class BitwiseReductionOp;
class Receive;
class Send;
class NaryOp;
class BitSlice;
class DynamicBitSlice;
class BitSliceUpdate;
class CompareOp;
class Concat;
class CountedFor;
class DynamicCountedFor;
class ExtendOp;
class Invoke;
class Literal;
class Map;
class OneHot;
class OneHotSelect;
class PrioritySelect;
class Param;
class Next;
class Select;
class Tuple;
class TupleIndex;
class UnOp;
class Decode;
class Encode;
class InputPort;
class OutputPort;
class RegisterRead;
class RegisterWrite;
class InstantiationOutput;
class InstantiationInput;
class Gate;

// Returns whether the given Op has the OpT node subclass.
class Node;

template <typename OpT>
bool IsOpClass(Op op) {
  static_assert(std::is_base_of<Node, OpT>::value,
                "OpT is not a Node subclass");
  return false;
}

template <>
bool IsOpClass<AfterAll>(Op op);
template <>
bool IsOpClass<MinDelay>(Op op);
template <>
bool IsOpClass<Array>(Op op);
template <>
bool IsOpClass<ArrayIndex>(Op op);
template <>
bool IsOpClass<ArraySlice>(Op op);
template <>
bool IsOpClass<ArrayUpdate>(Op op);
template <>
bool IsOpClass<ArrayConcat>(Op op);
template <>
bool IsOpClass<BinOp>(Op op);
template <>
bool IsOpClass<ArithOp>(Op op);
template <>
bool IsOpClass<PartialProductOp>(Op op);
template <>
bool IsOpClass<Assert>(Op op);
template <>
bool IsOpClass<Trace>(Op op);
template <>
bool IsOpClass<Cover>(Op op);
template <>
bool IsOpClass<BitwiseReductionOp>(Op op);
template <>
bool IsOpClass<Receive>(Op op);
template <>
bool IsOpClass<Send>(Op op);
template <>
bool IsOpClass<NaryOp>(Op op);
template <>
bool IsOpClass<BitSlice>(Op op);
template <>
bool IsOpClass<DynamicBitSlice>(Op op);
template <>
bool IsOpClass<BitSliceUpdate>(Op op);
template <>
bool IsOpClass<CompareOp>(Op op);
template <>
bool IsOpClass<Concat>(Op op);
template <>
bool IsOpClass<CountedFor>(Op op);
template <>
bool IsOpClass<DynamicCountedFor>(Op op);
template <>
bool IsOpClass<ExtendOp>(Op op);
template <>
bool IsOpClass<Invoke>(Op op);
template <>
bool IsOpClass<Literal>(Op op);
template <>
bool IsOpClass<Map>(Op op);
template <>
bool IsOpClass<OneHot>(Op op);
template <>
bool IsOpClass<OneHotSelect>(Op op);
template <>
bool IsOpClass<PrioritySelect>(Op op);
template <>
bool IsOpClass<Param>(Op op);
template <>
bool IsOpClass<Next>(Op op);
template <>
bool IsOpClass<Select>(Op op);
template <>
bool IsOpClass<Tuple>(Op op);
template <>
bool IsOpClass<TupleIndex>(Op op);
template <>
bool IsOpClass<UnOp>(Op op);
template <>
bool IsOpClass<Decode>(Op op);
template <>
bool IsOpClass<Encode>(Op op);
template <>
bool IsOpClass<InputPort>(Op op);
template <>
bool IsOpClass<OutputPort>(Op op);
template <>
bool IsOpClass<RegisterRead>(Op op);
template <>
bool IsOpClass<RegisterWrite>(Op op);
template <>
bool IsOpClass<InstantiationOutput>(Op op);
template <>
bool IsOpClass<InstantiationInput>(Op op);
template <>
bool IsOpClass<Gate>(Op op);

// Streams the string for "op" to the given output stream.
std::ostream& operator<<(std::ostream& os, Op op);

}  // namespace xls

#endif  // XLS_IR_OP_H_
