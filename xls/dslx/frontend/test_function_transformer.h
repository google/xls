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

#ifndef XLS_DSLX_FRONTEND_TEST_FUNCTION_TRANSFORMER_H_
#define XLS_DSLX_FRONTEND_TEST_FUNCTION_TRANSFORMER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

class TestFunctionTransformer {
 public:
  TestFunctionTransformer(Module& module, TypeInfo& ti)
      : source_module_(module), type_info_(ti) {}

  // Transforms test functions that spawn procs into TestProcs in the cloned
  // module. The original functions will be removed from the cloned module.
  absl::StatusOr<std::unique_ptr<Module>> TransformTestFunctions();

 private:
  // Transforms a single test function `fn` that spawns procs into a TestProc
  // (represented as an `Impl` block) in `new_module`.
  //
  // This involves partitioning statements, promoting variables used across
  // setup/runtime boundaries to proc fields, and generating the `new` and
  // `next` functions.
  absl::StatusOr<Impl*> TransformToTestProc(const Function& fn,
                                            Module& new_module);

  struct PromotionResult {
    absl::flat_hash_map<const NameDef*, std::string> original_to_unique_name;
    std::vector<const NameDef*> sorted_defs_to_promote;
  };

  // Promotes the given local variables to fields on the proc_def.
  //
  // Handles name uniquification if collisions occur, adds member nodes to
  // `proc_def`. Returns the mapping of original name defs to their new
  // unique names.
  absl::StatusOr<PromotionResult> PromoteVariablesToFields(
      const absl::flat_hash_set<const NameDef*>& defs_to_promote,
      Module& new_module, ProcDef& proc_def);

  struct ClonedSetup {
    std::vector<Statement*> cloned_statements;
    absl::flat_hash_map<const NameDef*, NameDef*> original_to_cloned_local_def;
  };

  // Clones the setup (constexpr) statements into `new_module` and along the way
  // renames locals to their "promoted" equivalent field.
  //
  // Returns the cloned statements and a mapping from original name defs to
  // their cloned local definitions.
  absl::StatusOr<ClonedSetup> CloneSetupBlock(
      const Function& fn, const std::vector<Statement*>& const_statements,
      const PromotionResult& promotion_result, Module& new_module);

  // Synthesizes the `new` function for the proc.
  absl::StatusOr<Function*> CreateNewFunction(
      const Function& fn, Module& new_module, TypeAnnotation* proc_def_type,
      TypeAnnotation* terminator_channel_type, const PromotionResult& promotion,
      const ClonedSetup& setup);

  // Returns true if the given statement wraps a known constant expression.
  bool IsConstExpr(const Statement* stmt);

  const Module& source_module_;
  TypeInfo& type_info_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_TEST_FUNCTION_TRANSFORMER_H_
