// Copyright 2025 The XLS Authors
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

#include "xls/visualization/math_notation.h"

#include <functional>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"

namespace xls {

std::string ToMathNotation(const Node* node) {
  return ToMathNotation(node, [](const Node*) { return false; });
}

std::string ToMathNotation(const Node* node,
                           std::function<bool(const Node*)> treat_as_atom) {
  if (!node) {
    return "";
  }
  if (treat_as_atom(node)) {
    return node->GetName();
  }

  absl::Span<Node* const> operands = node->operands();
  std::vector<std::string> operand_strings;
  operand_strings.reserve(operands.size());
  for (Node* operand : operands) {
    operand_strings.push_back(ToMathNotation(operand, treat_as_atom));
  }
  std::string lhs = operands.empty() ? "" : operand_strings[0];
  std::string rhs = operands.size() < 2 ? "" : operand_strings[1];

  switch (node->op()) {
    case Op::kLiteral:
      return node->As<Literal>()->value().ToString();
    case Op::kNot:
      return absl::StrCat("!", lhs);
    case Op::kNeg:
      return absl::StrCat("-", lhs);
    case Op::kAnd:
      return absl::StrCat("(", absl::StrJoin(operand_strings, " & "), ")");
    case Op::kOr:
      return absl::StrCat("(", absl::StrJoin(operand_strings, " | "), ")");
    case Op::kNand:
      return absl::StrCat("!(", absl::StrJoin(operand_strings, " & "), ")");
    case Op::kNor:
      return absl::StrCat("!(", absl::StrJoin(operand_strings, " | "), ")");
    case Op::kXor:
      return absl::StrCat("(", absl::StrJoin(operand_strings, " ⊕ "), ")");
    case Op::kConcat:
      return absl::StrCat("(", absl::StrJoin(operand_strings, " ++ "), ")");
    case Op::kBitSlice:
      return absl::StrCat(
          "(", lhs, "[", node->As<BitSlice>()->start(), ":",
          node->As<BitSlice>()->start() + node->As<BitSlice>()->width(), "]",
          ")");
    case Op::kAdd:
      return absl::StrCat("(", lhs, " + ", rhs, ")");
    case Op::kSub:
      return absl::StrCat("(", lhs, " - ", rhs, ")");
    case Op::kUMul:
    case Op::kSMul:
      return absl::StrCat("(", lhs, " * ", rhs, ")");
    case Op::kUDiv:
    case Op::kSDiv:
      return absl::StrCat("(", lhs, " / ", rhs, ")");
    case Op::kUMod:
    case Op::kSMod:
      return absl::StrCat("(", lhs, " % ", rhs, ")");
    case Op::kEq:
      return absl::StrCat("(", lhs, " == ", rhs, ")");
    case Op::kNe:
      return absl::StrCat("(", lhs, " != ", rhs, ")");
    case Op::kULt:
    case Op::kSLt:
      return absl::StrCat("(", lhs, " < ", rhs, ")");
    case Op::kUGt:
    case Op::kSGt:
      return absl::StrCat("(", lhs, " > ", rhs, ")");
    case Op::kULe:
    case Op::kSLe:
      return absl::StrCat("(", lhs, " <= ", rhs, ")");
    case Op::kUGe:
    case Op::kSGe:
      return absl::StrCat("(", lhs, " >= ", rhs, ")");
    case Op::kShll:
      return absl::StrCat("(", lhs, " << ", rhs, ")");
    case Op::kShra:
      return absl::StrCat("(", lhs, " >>> ", rhs, ")");
    case Op::kShrl:
      return absl::StrCat("(", lhs, " >> ", rhs, ")");
    default:
      return node->GetName();
  }
}

}  // namespace xls
