// Copyright 2021 The XLS Authors
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

#ifndef XLS_DSLX_IR_CONVERT_EXTRACT_CONVERSION_ORDER_H_
#define XLS_DSLX_IR_CONVERT_EXTRACT_CONVERSION_ORDER_H_

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/proc.h"
#include "xls/dslx/frontend/proc_id.h"
#include "xls/dslx/ir_convert/conversion_record.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

// Describes a callee function in the conversion order (see
// ConversionRecord::callees).
class Callee {
 public:
  // Proc definitions can't be directly translated into IR: they're always
  // instantiated based on a Spawn or series thereof.
  // `proc_id` holds the series of spawned procs leading up to this callee.
  //
  // This is conceptually similar to the instantiation of parametric functions,
  // except that even non-parametric Procs need instantiation details to be
  // converted to IR.
  static absl::StatusOr<Callee> Make(Function* f, const Invocation* invocation,
                                     Module* m, TypeInfo* type_info,
                                     ParametricEnv parametric_env,
                                     std::optional<ProcId> proc_id);

  Function* f() const { return f_; }
  const Invocation* invocation() const { return invocation_; }
  Module* m() const { return m_; }
  TypeInfo* type_info() const { return type_info_; }
  const ParametricEnv& parametric_env() const { return parametric_env_; }
  // If nullopt is returned, that means that this isn't a proc function.
  const std::optional<ProcId>& proc_id() const { return proc_id_; }
  std::string ToString() const;

 private:
  Callee(Function* f, const Invocation* invocation, Module* m,
         TypeInfo* type_info, ParametricEnv parametric_env,
         std::optional<ProcId> proc_id);

  Function* f_;
  const Invocation* invocation_;
  Module* m_;
  TypeInfo* type_info_;
  ParametricEnv parametric_env_;
  std::optional<ProcId> proc_id_;
};

// Returns (topological) order for functions to be converted to IR.
//
// Returned order should be deterministic, since the call graph is traversed in
// a deterministic way. The top-level procs are in the order that they appear in
// the module. Sub-procs are in the order that they are invoked within a proc.
//
// Args:
//  module: Module to convert the (non-parametric) functions for.
//  type_info: Mapping from node to type.
//  include_tests: should test-functions be included.
absl::StatusOr<std::vector<ConversionRecord>> GetOrder(
    Module* module, TypeInfo* type_info, bool include_tests = false);

// Returns a reverse topological order for functions to be converted to IR given
// "f" as the entry function.
//
// Returned order should be deterministic, since the call graph is traversed in
// a deterministic way.
//
// Args:
//  f: The top level function.
//  type_info: Mapping from node to type.
absl::StatusOr<std::vector<ConversionRecord>> GetOrderForEntry(
    std::variant<Function*, Proc*> entry, TypeInfo* type_info);

// Top level procs are procs where their config or next function is not invoked
// within the module.
//
// Note that parametric procs must be instantiated, and thus are never
// top-level.
//
// For example, for a given module with four procs: ProcA, ProcB, ProcC and
// ProcD. The procs have the following invocation scheme:
// * ProcA invokes ProcC and ProcD, and
// * ProcB does not invoke any procs.
// Given that there is no invocation of ProcA and ProcB, they (ProcA and ProcB)
// are the top level procs.
absl::StatusOr<std::vector<Proc*>> GetTopLevelProcs(Module* module,
                                                    TypeInfo* type_info,
                                                    bool include_tests = false);

}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERT_EXTRACT_CONVERSION_ORDER_H_
