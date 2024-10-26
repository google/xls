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
#include <type_traits>

#include "absl/status/statusor.h"
#include "xls/ir/op.pb.h"
#include "xls/ir/op_list.h"

// TODO(meheff): Add comments to classes and methods.

namespace xls {

// Enumerates the operator for nodes in the IR.
enum class Op : int8_t {
#define MAKE_ENUM(name, a, b, c) name,
  XLS_FOR_EACH_OP_TYPE(MAKE_ENUM)
#undef MAKE_ENUM
};

// List of all the operators for nodes in the IR.
inline constexpr auto kAllOps = std::to_array<Op>({
#define MAKE_ENUM_REF(name, a, b, c) Op::name,
    XLS_FOR_EACH_OP_TYPE(MAKE_ENUM_REF)
#undef MAKE_ENUM_REF
});

// Converts an OpProto into an Op.
Op FromOpProto(OpProto op_proto);

// Converts an Op into an OpProto.
OpProto ToOpProto(Op op);

// Converts the "op" enumeration to a human readable string.
// TODO(allight): We return a string literal so this can be a string_view
// return. Need to update all the users.
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

// Forward declaration for base Node class
class Node;

// Returns whether the given Op has the OpT node subclass.
template <typename OpT>
constexpr bool IsOpClass(Op op) {
  static_assert(std::is_base_of<Node, OpT>::value && !std::is_same_v<OpT, Node>,
                "OpT is not a Node subclass");
  // Return true if op is one of the elements of OpT::kOps
  for (auto it = OpT::kOps.begin(); it != OpT::kOps.end(); ++it) {
    if (*it == op) {
      return true;
    }
  }
  return false;
}

// Streams the string for "op" to the given output stream.
std::ostream& operator<<(std::ostream& os, Op op);

}  // namespace xls

#endif  // XLS_IR_OP_H_
