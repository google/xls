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
#include "xls/ir/op.pb.h"
#include "xls/ir/op_list.h"

namespace xls {

Op FromOpProto(OpProto op_proto) {
  switch (op_proto) {
    case OP_INVALID:
      LOG(FATAL) << "OP_INVALID received";
      break;
#define FROM_OP_PROTO(name, proto, a, b) \
  case proto:                            \
    return Op::name;
      XLS_FOR_EACH_OP_TYPE(FROM_OP_PROTO)
#undef FROM_OP_PROTO
    default:
      LOG(FATAL) << "Invalid OpProto: " << static_cast<int64_t>(op_proto);
  }
}

OpProto ToOpProto(Op op) {
  switch (op) {
#define TO_OP_PROTO(name, proto, a, b) \
  case Op::name:                       \
    return proto;
    XLS_FOR_EACH_OP_TYPE(TO_OP_PROTO)
#undef TO_OP_PROTO
  }
  LOG(FATAL) << "Invalid Op: " << static_cast<int64_t>(op);
}

// TODO(allight): This can be a string_view.
std::string OpToString(Op op) {
  switch (op) {
#define TO_OP_STRING(name, a, str, b) \
  case Op::name:                      \
    return str;
    XLS_FOR_EACH_OP_TYPE(TO_OP_STRING)
#undef TO_OP_STRING
  }
}

absl::StatusOr<Op> StringToOp(std::string_view op_str) {
  static const absl::NoDestructor<absl::flat_hash_map<std::string, Op>>
      string_map({
#define FROM_OP_STRING(name, a, str, b) {str, Op::name},
          XLS_FOR_EACH_OP_TYPE(FROM_OP_STRING)
#undef FROM_OP_STRING
      });
  auto found = string_map->find(op_str);
  if (found == string_map->end()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unknown operation for string-to-op conversion: ", op_str));
  }
  return found->second;
}
namespace {
constexpr int8_t GetOpTypes(Op op) {
  switch (op) {
#define TO_OP_TY(name, a, b, ty) \
  case Op::name:                 \
    return ty;
    XLS_FOR_EACH_OP_TYPE(TO_OP_TY)
#undef TO_OP_TY
  }
}
}  // namespace

bool OpIsCompare(Op op) {
  return (GetOpTypes(op) & op_types::kComparison) == op_types::kComparison;
}

bool OpIsAssociative(Op op) {
  return (GetOpTypes(op) & op_types::kAssociative) == op_types::kAssociative;
}

bool OpIsCommutative(Op op) {
  return (GetOpTypes(op) & op_types::kCommutative) == op_types::kCommutative;
}

bool OpIsBitWise(Op op) {
  return (GetOpTypes(op) & op_types::kBitWise) == op_types::kBitWise;
}

bool OpIsSideEffecting(Op op) {
  return (GetOpTypes(op) & op_types::kSideEffecting) ==
         op_types::kSideEffecting;
}

std::ostream& operator<<(std::ostream& os, Op op) {
  os << OpToString(op);
  return os;
}

}  // namespace xls
