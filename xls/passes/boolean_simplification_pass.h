// Copyright 2020 The XLS Authors
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

#ifndef XLS_PASSES_BOOLEAN_SIMPLIFICATION_PASS_H_
#define XLS_PASSES_BOOLEAN_SIMPLIFICATION_PASS_H_

#include <cstdint>
#include <optional>
#include <string_view>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace internal {

// Exposed in the header for testing.
class TruthTable {
 public:
  // xyz_present and xyz_negated must be bit_count == 3 for each of the operands
  // on the input frontier (see .cc file for details). A value like xyz_present
  // == 0b100 indicates 'x' is present, 0b011 indicates 'y' and 'z' are present,
  // etc.
  TruthTable(const Bits& xyz_present, const Bits& xyz_negated,
             std::optional<Op> logical_op);

  // Computes the result vector for this operation, as specified in the
  // constructor.
  Bits ComputeTruthTable() const;

  bool MatchesVector(const Bits& table) const;

  // Returns whether the original node matches this logical function with the
  // given operands (note all logical operations we express in this way are
  // symmetrical with respect to permutations in the input operands).
  bool MatchesSymmetrical(Node* original,
                          absl::Span<const Node* const> operands) const;

  // Creates a replacement node to use in lieu of the original that corresponds
  // to this truth table with the given input frontier operands.
  absl::StatusOr<Node*> CreateReplacement(const SourceInfo& original_loc,
                                          absl::Span<Node* const> operands,
                                          FunctionBase* f) const;

  // Gets the truth table (input) vector for operand "i".
  static Bits GetInitialVector(int64_t i);
  static Bits RunNaryOp(Op op, absl::Span<const Bits> operands);

 private:
  Bits xyz_present_;
  Bits xyz_negated_;
  std::optional<Op> logical_op_;
};

}  // namespace internal

// Attempts to simplify bitwise / boolean expressions (e.g. of multiple
// variables).
class BooleanSimplificationPass : public OptimizationFunctionBasePass {
 public:
  static constexpr std::string_view kName = "bool_simp";
  BooleanSimplificationPass()
      : OptimizationFunctionBasePass(kName, "boolean simplification") {}

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext* context) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_BOOLEAN_SIMPLIFICATION_PASS_H_
