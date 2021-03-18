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

// Describes a callee function in the conversion order (see
// ConversionRecord::callees).
class Callee {
 public:
  Callee(Function* f, Module* m, TypeInfo* type_info,
         SymbolicBindings sym_bindings);

  Function* f() const { return f_; }
  Module* m() const { return m_; }
  TypeInfo* type_info() const { return type_info_; }
  const SymbolicBindings& sym_bindings() const { return sym_bindings_; }
  std::string ToString() const;

 private:
  Function* f_;
  Module* m_;
  TypeInfo* type_info_;
  SymbolicBindings sym_bindings_;
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
      Function* f, Module* module, TypeInfo* type_info,
      SymbolicBindings symbolic_bindings, std::vector<Callee> callees);

  // Integrity-checks that the symbolic_bindings provided are sufficient to
  // instantiate f (i.e. if it is parametric). Returns an internal error status
  // if they are not sufficient.
  static absl::Status ValidateParametrics(
      Function* f, const SymbolicBindings& symbolic_bindings);

  Function* f() const { return f_; }
  Module* module() const { return module_; }
  TypeInfo* type_info() const { return type_info_; }
  const SymbolicBindings& symbolic_bindings() const {
    return symbolic_bindings_;
  }
  const std::vector<Callee>& callees() const { return callees_; }

  std::string ToString() const;

 private:
  ConversionRecord(Function* f, Module* module, TypeInfo* type_info,
                   SymbolicBindings symbolic_bindings,
                   std::vector<Callee> callees)
      : f_(f),
        module_(module),
        type_info_(type_info),
        symbolic_bindings_(std::move(symbolic_bindings)),
        callees_(std::move(callees)) {}

  Function* f_;
  Module* module_;
  TypeInfo* type_info_;
  SymbolicBindings symbolic_bindings_;
  std::vector<Callee> callees_;
};

// Returns (topological) order for functions to be converted to IR.
//
// Returned order should be deterministic, since the call graph is traversed in
// a deterministic way.
//
// Args:
//  module: Module to convert the (non-parametric) functions for.
//  type_info: Mapping from node to type.
//  imports: Transitive imports that are required by "module".
//  traverse_tests: Whether to traverse DSLX test constructs. This flag should
//    be set if we intend to run functions only called from test constructs
//    through the JIT.
absl::StatusOr<std::vector<ConversionRecord>> GetOrder(
    Module* module, TypeInfo* type_info, bool traverse_tests = false);

}  // namespace xls::dslx

#endif  // XLS_DSLX_CPP_EXTRACT_CONVERSION_ORDER_H_
