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

#ifndef XLS_IR_OP_LIST_H_
#define XLS_IR_OP_LIST_H_

#include <cstdint>
namespace xls {

namespace op_types {
inline constexpr uint8_t kStandard = 0b00000000;
inline constexpr uint8_t kComparison = 0b00000001;
inline constexpr uint8_t kAssociative = 0b00000010;
inline constexpr uint8_t kCommutative = 0b00000100;
inline constexpr uint8_t kBitWise = 0b00001000;
inline constexpr uint8_t kSideEffecting = 0b00010000;
}  // namespace op_types

// Run a macro for each op type.
// Macro args are F(kEnumName, PROTO_NAME, "ir_string", type_bitmap)
#define XLS_FOR_EACH_OP_TYPE(F)                                                \
  F(kAdd, OP_ADD, "add", op_types::kAssociative | xls::op_types::kCommutative) \
  F(kAfterAll, OP_AFTER_ALL, "after_all",                                      \
    op_types::kAssociative | op_types::kCommutative)                           \
  F(kAnd, OP_AND, "and",                                                       \
    op_types::kAssociative | op_types::kCommutative | op_types::kBitWise)      \
  F(kAndReduce, OP_AND_REDUCE, "and_reduce", op_types::kStandard)              \
  F(kArray, OP_ARRAY, "array", op_types::kStandard)                            \
  F(kArrayConcat, OP_ARRAY_CONCAT, "array_concat", op_types::kStandard)        \
  F(kArrayIndex, OP_ARRAY_INDEX, "array_index", op_types::kStandard)           \
  F(kArraySlice, OP_ARRAY_SLICE, "array_slice", op_types::kStandard)           \
  F(kArrayUpdate, OP_ARRAY_UPDATE, "array_update", op_types::kStandard)        \
  F(kAssert, OP_ASSERT, "assert", op_types::kSideEffecting)                    \
  F(kBitSlice, OP_BIT_SLICE, "bit_slice", op_types::kStandard)                 \
  F(kBitSliceUpdate, OP_BIT_SLICE_UPDATE, "bit_slice_update",                  \
    op_types::kStandard)                                                       \
  F(kConcat, OP_CONCAT, "concat", op_types::kStandard)                         \
  F(kCountedFor, OP_COUNTED_FOR, "counted_for", op_types::kStandard)           \
  F(kCover, OP_COVER, "cover", op_types::kSideEffecting)                       \
  F(kDecode, OP_DECODE, "decode", op_types::kStandard)                         \
  F(kDynamicBitSlice, OP_DYNAMIC_BIT_SLICE, "dynamic_bit_slice",               \
    op_types::kStandard)                                                       \
  F(kDynamicCountedFor, OP_DYNAMIC_COUNTED_FOR, "dynamic_counted_for",         \
    op_types::kStandard)                                                       \
  F(kEncode, OP_ENCODE, "encode", op_types::kStandard)                         \
  F(kEq, OP_EQ, "eq", op_types::kComparison | op_types::kCommutative)          \
  F(kGate, OP_GATE, "gate", op_types::kSideEffecting)                          \
  F(kIdentity, OP_IDENTITY, "identity", op_types::kStandard)                   \
  F(kInputPort, OP_INPUT_PORT, "input_port", op_types::kSideEffecting)         \
  F(kInstantiationInput, OP_INSTANTIATION_INPUT, "instantiation_input",        \
    op_types::kSideEffecting)                                                  \
  F(kInstantiationOutput, OP_INSTANTIATION_OUTPUT, "instantiation_output",     \
    op_types::kSideEffecting)                                                  \
  F(kInvoke, OP_INVOKE, "invoke", op_types::kStandard)                         \
  F(kLiteral, OP_LITERAL, "literal", op_types::kStandard)                      \
  F(kMap, OP_MAP, "map", op_types::kStandard)                                  \
  F(kMinDelay, OP_MIN_DELAY, "min_delay", op_types::kStandard)                 \
  F(kNand, OP_NAND, "nand", op_types::kBitWise | op_types::kCommutative)       \
  F(kNe, OP_NE, "ne", op_types::kComparison | op_types::kCommutative)          \
  F(kNeg, OP_NEG, "neg", op_types::kStandard)                                  \
  F(kNewChannel, OP_NEW_CHANNEL, "new_channel", op_types::kSideEffecting)      \
  F(kNext, OP_NEXT_VALUE, "next_value", op_types::kSideEffecting)              \
  F(kNor, OP_NOR, "nor", op_types::kBitWise | op_types::kCommutative)          \
  F(kNot, OP_NOT, "not", op_types::kBitWise)                                   \
  F(kOneHot, OP_ONE_HOT, "one_hot", op_types::kStandard)                       \
  F(kOneHotSel, OP_ONE_HOT_SEL, "one_hot_sel", op_types::kStandard)            \
  F(kOr, OP_OR, "or",                                                          \
    op_types::kBitWise | op_types::kCommutative | op_types::kAssociative)      \
  F(kOrReduce, OP_OR_REDUCE, "or_reduce", op_types::kStandard)                 \
  F(kOutputPort, OP_OUTPUT_PORT, "output_port", op_types::kSideEffecting)      \
  F(kParam, OP_PARAM, "param", op_types::kSideEffecting)                       \
  F(kPrioritySel, OP_PRIORITY_SEL, "priority_sel", op_types::kStandard)        \
  F(kReceive, OP_RECEIVE, "receive", op_types::kSideEffecting)                 \
  F(kRecvChannelEnd, OP_RECV_CHANNEL_END, "recv_channel_end",                  \
    op_types::kSideEffecting)                                                  \
  F(kRegisterRead, OP_REGISTER_READ, "register_read",                          \
    op_types::kSideEffecting)                                                  \
  F(kRegisterWrite, OP_REGISTER_WRITE, "register_write",                       \
    op_types::kSideEffecting)                                                  \
  F(kReverse, OP_REVERSE, "reverse", op_types::kStandard)                      \
  F(kSDiv, OP_SDIV, "sdiv", op_types::kStandard)                               \
  F(kSGe, OP_SGE, "sge", op_types::kComparison)                                \
  F(kSGt, OP_SGT, "sgt", op_types::kComparison)                                \
  F(kSLe, OP_SLE, "sle", op_types::kComparison)                                \
  F(kSLt, OP_SLT, "slt", op_types::kComparison)                                \
  F(kSMod, OP_SMOD, "smod", op_types::kStandard)                               \
  F(kSMul, OP_SMUL, "smul", op_types::kAssociative | op_types::kCommutative)   \
  F(kSMulp, OP_SMULP, "smulp",                                                 \
    op_types::kAssociative | op_types::kCommutative)                           \
  F(kSel, OP_SEL, "sel", op_types::kStandard)                                  \
  F(kSend, OP_SEND, "send", op_types::kSideEffecting)                          \
  F(kSendChannelEnd, OP_SEND_CHANNEL_END, "send_channel_end",                  \
    op_types::kSideEffecting)                                                  \
  F(kShll, OP_SHLL, "shll", op_types::kStandard)                               \
  F(kShra, OP_SHRA, "shra", op_types::kStandard)                               \
  F(kShrl, OP_SHRL, "shrl", op_types::kStandard)                               \
  F(kSignExt, OP_SIGN_EXT, "sign_ext", op_types::kStandard)                    \
  F(kStateRead, OP_STATE_READ, "state_read", op_types::kSideEffecting)         \
  F(kSub, OP_SUB, "sub", op_types::kStandard)                                  \
  F(kTrace, OP_TRACE, "trace", op_types::kSideEffecting)                       \
  F(kTuple, OP_TUPLE, "tuple", op_types::kStandard)                            \
  F(kTupleIndex, OP_TUPLE_INDEX, "tuple_index", op_types::kStandard)           \
  F(kUDiv, OP_UDIV, "udiv", op_types::kStandard)                               \
  F(kUGe, OP_UGE, "uge", op_types::kComparison)                                \
  F(kUGt, OP_UGT, "ugt", op_types::kComparison)                                \
  F(kULe, OP_ULE, "ule", op_types::kComparison)                                \
  F(kULt, OP_ULT, "ult", op_types::kComparison)                                \
  F(kUMod, OP_UMOD, "umod", op_types::kStandard)                               \
  F(kUMul, OP_UMUL, "umul", op_types::kAssociative | op_types::kCommutative)   \
  F(kUMulp, OP_UMULP, "umulp",                                                 \
    op_types::kAssociative | op_types::kCommutative)                           \
  F(kXor, OP_XOR, "xor",                                                       \
    op_types::kBitWise | op_types::kAssociative | op_types::kCommutative)      \
  F(kXorReduce, OP_XOR_REDUCE, "xor_reduce", op_types::kStandard)              \
  F(kZeroExt, OP_ZERO_EXT, "zero_ext", op_types::kStandard)

}  // namespace xls

#endif  // XLS_IR_OP_LIST_H_
