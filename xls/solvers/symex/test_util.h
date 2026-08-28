// Copyright 2026 The XLS Authors
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

#ifndef XLS_SOLVERS_SYMEX_TEST_UTIL_H_
#define XLS_SOLVERS_SYMEX_TEST_UTIL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/types/span.h"
#include "xls/ir/ir_test_base.h"
#include "xls/solvers/symex/symbolic_path.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {

// Base fixture for symbolic execution unit tests managing Z3 context lifecycle.
class SymExTestBase : public IrTestBase {
 protected:
  void SetUp() override {
    IrTestBase::SetUp();
    config_ = Z3_mk_config();
    ctx_ = Z3_mk_context(config_);
  }

  void TearDown() override {
    if (ctx_ != nullptr) {
      Z3_del_context(ctx_);
    }
    if (config_ != nullptr) {
      Z3_del_config(config_);
    }
    IrTestBase::TearDown();
  }

  Z3_config config_ = nullptr;
  Z3_context ctx_ = nullptr;
};

// Returns true if cond1 and cond2 are mutually exclusive (i.e. check(cond1 &&
// cond2) == UNSAT).
bool AreMutuallyExclusive(Z3_context ctx, Z3_ast cond1, Z3_ast cond2);

// Returns true if the disjunction of all path conditions covers 100% of the
// input domain (i.e. check(!combined_path_condition) == UNSAT).
bool IsExhaustiveCoverage(Z3_context ctx, absl::Span<const SymbolicPath> paths);

// Matcher for BranchDecision checking arm_index and is_default.
MATCHER_P2(BranchDecisionIs, arm_index, is_default,
           absl::StrCat("has arm_index ", arm_index, " and is_default ",
                        is_default ? "true" : "false")) {
  return arg.arm_index == arm_index && arg.is_default() == is_default;
}

// Matcher for a single-decision SymbolicPath checking its first branch
// decision.
MATCHER_P2(SymbolicPathIs, arm_index, is_default,
           absl::StrCat("has first decision arm_index ", arm_index,
                        " and is_default ", is_default ? "true" : "false")) {
  if (arg.branch_decisions.empty()) {
    *result_listener << "has empty branch_decisions";
    return false;
  }
  const auto& decision = arg.branch_decisions[0];
  return decision.arm_index == arm_index && decision.is_default() == is_default;
}

}  // namespace xls::solvers::symex

#endif  // XLS_SOLVERS_SYMEX_TEST_UTIL_H_
