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
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/proc.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

// ProcId is used to represent a unique instantiation of a Proc, that is, to
// differentiate the instance of Proc Foo spawned from Proc Bar from the one
// spawned from Proc Baz. Each instance can have different member data:
// different constants or channels, so we need to be able to identify each
// separately.
struct ProcId {
  // Contains the "spawn chain": the series of Procs through which this Proc was
  // spawned, with the oldest/"root" proc as element 0.  Contains the current
  // proc, as well. The second element of each pair is a zero-based spawn index
  // of that same proc by the spawning proc. For example, with a spawn chain
  // like:
  //     A |-> D |-> B -> C
  //       |-> B -> C
  //       |-> B -> C
  //       |-> E |-> B -> C
  //
  // the `proc_instance_stack` for each `C` would look like:
  //    [{A, 0}, {D, 0}, {B, 0}, {C, 0}]
  //    [{A, 0}, {B, 0}, {C, 0}]
  //    [{A, 0}, {B, 1}, {C, 0}]
  //    [{A, 0}, {E, 0}, {B, 0}, {C, 0}]
  std::vector<std::pair<Proc*, int>> proc_instance_stack;

  std::string ToString() const {
    if (proc_instance_stack.empty()) {
      return "";
    }
    // The first proc in a chain never needs an instance count. Leaving it out
    // specifically when the chain length is more than 1 gets us the historical
    // output in most cases (where there was only an instance count at the end
    // of the chain).
    CHECK_EQ(proc_instance_stack[0].second, 0);
    const bool omit_first_instance_count = proc_instance_stack.size() > 1;
    std::string part_with_instance_counts = absl::StrJoin(
        proc_instance_stack.begin() + (omit_first_instance_count ? 1 : 0),
        proc_instance_stack.end(), "->",
        [](std::string* out, const std::pair<Proc*, int> p) {
          absl::StrAppendFormat(out, "%s:%d", p.first->identifier(), p.second);
        });
    return omit_first_instance_count
               ? absl::StrCat(proc_instance_stack[0].first->identifier(), "->",
                              part_with_instance_counts)
               : part_with_instance_counts;
  }

  bool operator==(const ProcId& other) const {
    return proc_instance_stack == other.proc_instance_stack;
  }

  template <typename H>
  friend H AbslHashValue(H h, const ProcId& pid) {
    return H::combine(std::move(h), pid.proc_instance_stack);
  }
};

// An object that deals out `ProcId` instances.
class ProcIdFactory {
 public:
  // Creates a `ProcId` representing the given `spawnee` spawned by the given
  // `parent` context. If `count_as_new_instance` is true, then subsequent calls
  // with the same `parent` and `spawnee` will get a new instance count value.
  // Otherwise, subsequent calls will get an equivalent `ProcId` to the one
  // returned by this call.
  ProcId CreateProcId(const ProcId& parent, Proc* spawnee,
                      bool count_as_new_instance = true);

 private:
  // Maps each `parent` and `spawnee` identifier passed to `CreateProcId` to the
  // number of instances of that pairing, i.e., the number of times that
  // `parent` and `spawnee` have been passed in with `true` for
  // `count_as_new_instance`.
  absl::flat_hash_map<std::pair<ProcId, std::string>, int> instance_counts_;
};

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

// Record used in sequence, noting order functions should be converted in.
//
// Describes a function instance that should be emitted (in an order determined
// by an encapsulating sequence). Annotated with metadata that describes the
// call graph instance.
//
// Attributes:
//   f: Function AST node to convert.
//   module: Module that f resides in.
//   type_info: Node to type mapping for use in converting this
//     function instance.
//   callees: Function names that 'f' calls.
//   parametric_env: Parametric bindings for this function instance.
//   callees: Functions that this instance calls.
class ConversionRecord {
 public:
  // Note: performs ValidateParametrics() to potentially return an error status.
  static absl::StatusOr<ConversionRecord> Make(
      Function* f, const Invocation* invocation, Module* module,
      TypeInfo* type_info, ParametricEnv parametric_env,
      std::vector<Callee> callees, std::optional<ProcId> proc_id, bool is_top);

  // Integrity-checks that the parametric_env provided are sufficient to
  // instantiate f (i.e. if it is parametric). Returns an internal error status
  // if they are not sufficient.
  static absl::Status ValidateParametrics(Function* f,
                                          const ParametricEnv& parametric_env);

  Function* f() const { return f_; }
  const Invocation* invocation() const { return invocation_; }
  Module* module() const { return module_; }
  TypeInfo* type_info() const { return type_info_; }
  const ParametricEnv& parametric_env() const { return parametric_env_; }
  std::optional<ProcId> proc_id() const { return proc_id_; }
  bool IsTop() const { return is_top_; }

  std::string ToString() const;

 private:
  ConversionRecord(Function* f, const Invocation* invocation, Module* module,
                   TypeInfo* type_info, ParametricEnv parametric_env,
                   std::vector<Callee> callees, std::optional<ProcId> proc_id,
                   bool is_top)
      : f_(f),
        invocation_(invocation),
        module_(module),
        type_info_(type_info),
        parametric_env_(std::move(parametric_env)),
        callees_(std::move(callees)),
        proc_id_(std::move(proc_id)),
        is_top_(is_top) {}

  Function* f_;
  const Invocation* invocation_;
  Module* module_;
  TypeInfo* type_info_;
  ParametricEnv parametric_env_;
  std::vector<Callee> callees_;
  std::optional<ProcId> proc_id_;
  bool is_top_;
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
absl::StatusOr<std::vector<ConversionRecord>> GetOrder(Module* module,
                                                       TypeInfo* type_info);

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
                                                    TypeInfo* type_info);

}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERT_EXTRACT_CONVERSION_ORDER_H_
