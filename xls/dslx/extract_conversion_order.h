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

#ifndef XLS_DSLX_CPP_EXTRACT_CONVERSION_ORDER_H_
#define XLS_DSLX_CPP_EXTRACT_CONVERSION_ORDER_H_

#include "absl/container/flat_hash_map.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/symbolic_bindings.h"
#include "xls/dslx/type_info.h"

namespace xls::dslx {

// ProcId is used to represent a unique instantiation of a Proc, that is, to
// differentiate the instance of Proc Foo spawned from Proc Bar from the one
// spawned from Proc Baz. Each instance can have different member data:
// different constants or channels, so we need to be able to identify each
// separately.
struct ProcId {
  // Contains the "spawn chain": the series of Procs through which this Proc was
  // spawned, with the oldest/"root" proc as element 0.  Contains the current
  // proc, as well.
  std::vector<Proc*> proc_stack;

  // The index of this Proc in the proc stack. If a Proc is spawned > 1 inside
  // the same Proc config function, this index differentiates the spawnees.
  // Each unique proc stack will have its count start at 0 - in other words,
  // the sequence:
  // spawn foo()(c0)
  // spawn bar()(c1)
  // spawn foo()(c2)
  // Would result in IDs of:
  // foo:0, bar:0, and foo:1, respectively.
  int instance;

  std::string ToString() const {
    return absl::StrCat(absl::StrJoin(proc_stack, "->",
                                      [](std::string* out, const Proc* p) {
                                        out->append(p->identifier());
                                      }),
                        ":", instance);
    return "";
  }

  bool operator==(const ProcId& other) const {
    return proc_stack == other.proc_stack && instance == other.instance;
  }

  template <typename H>
  friend H AbslHashValue(H h, const ProcId& pid) {
    return H::combine(std::move(h), pid.proc_stack, pid.instance);
  }
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
                                     SymbolicBindings sym_bindings,
                                     std::optional<ProcId> proc_id);

  bool IsFunction() const;
  Function* f() const { return f_; }
  const Invocation* invocation() const { return invocation_; }
  Module* m() const { return m_; }
  TypeInfo* type_info() const { return type_info_; }
  const SymbolicBindings& sym_bindings() const { return sym_bindings_; }
  // If nullopt is returned, that means that this isn't a proc function.
  const std::optional<ProcId>& proc_id() const { return proc_id_; }
  std::string ToString() const;

 private:
  Callee(Function* f, const Invocation* invocation, Module* m,
         TypeInfo* type_info, SymbolicBindings sym_bindings,
         std::optional<ProcId> proc_id);

  Function* f_;
  const Invocation* invocation_;
  Module* m_;
  TypeInfo* type_info_;
  SymbolicBindings sym_bindings_;
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
//   symbolic_bindings: Parametric bindings for this function instance.
//   callees: Functions that this instance calls.
class ConversionRecord {
 public:
  // Note: performs ValidateParametrics() to potentially return an error status.
  static absl::StatusOr<ConversionRecord> Make(
      Function* f, const Invocation* invocation, Module* module,
      TypeInfo* type_info, SymbolicBindings symbolic_bindings,
      std::vector<Callee> callees, std::optional<ProcId> proc_id, bool is_top);

  // Integrity-checks that the symbolic_bindings provided are sufficient to
  // instantiate f (i.e. if it is parametric). Returns an internal error status
  // if they are not sufficient.
  static absl::Status ValidateParametrics(
      Function* f, const SymbolicBindings& symbolic_bindings);

  Function* f() const { return f_; }
  const Invocation* invocation() const { return invocation_; }
  Module* module() const { return module_; }
  TypeInfo* type_info() const { return type_info_; }
  const SymbolicBindings& symbolic_bindings() const {
    return symbolic_bindings_;
  }
  const std::vector<Callee>& callees() const { return callees_; }
  std::optional<ProcId> proc_id() const { return proc_id_; }
  bool HasProcId() const { return proc_id_.has_value(); }
  bool IsTop() const { return is_top_; }

  std::string ToString() const;

 private:
  ConversionRecord(Function* f, const Invocation* invocation, Module* module,
                   TypeInfo* type_info, SymbolicBindings symbolic_bindings,
                   std::vector<Callee> callees, std::optional<ProcId> proc_id,
                   bool is_top)
      : f_(f),
        invocation_(invocation),
        module_(module),
        type_info_(type_info),
        symbolic_bindings_(std::move(symbolic_bindings)),
        callees_(std::move(callees)),
        proc_id_(proc_id),
        is_top_(is_top) {}

  Function* f_;
  const Invocation* invocation_;
  Module* module_;
  TypeInfo* type_info_;
  SymbolicBindings symbolic_bindings_;
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
//  traverse_tests: Whether to traverse DSLX test constructs. This flag should
//    be set if we intend to run functions only called from test constructs
//    through the JIT.
absl::StatusOr<std::vector<ConversionRecord>> GetOrder(
    Module* module, TypeInfo* type_info, bool traverse_tests = false);

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

}  // namespace xls::dslx

#endif  // XLS_DSLX_CPP_EXTRACT_CONVERSION_ORDER_H_
