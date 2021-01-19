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
#include "xls/dslx/cpp_ast.h"
#include "xls/dslx/symbolic_bindings.h"
#include "xls/dslx/type_info.h"

namespace xls::dslx {

// Describes a callee function in the conversion order (see
// ConversionRecord::callees).
struct Callee {
  Function* f;
  Module* m;
  std::shared_ptr<TypeInfo> type_info;
  SymbolicBindings sym_bindings;

  std::string ToString() const;
};

// Record used in sequence, noting order functions should be converted in.
//
// Describes a function instance that should be emitted (in an order determined
// by an encapsulating sequence). Annotated with metadata that describes the
// call graph instance.
//
// Attributes:
//   f: Function AST node to convert.
//   m: Module that f resides in.
//   type_info: Node to type mapping for use in converting this
//     function instance.
//   callees: Function names that 'f' calls.
//   bindings: Parametric bindings for this function instance.
//   callees: Functions that this instance calls.
struct ConversionRecord {
  Function* f;
  Module* m;
  std::shared_ptr<TypeInfo> type_info;
  SymbolicBindings bindings;
  std::vector<Callee> callees;

  std::string ToString() const;
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
    Module* module, const std::shared_ptr<TypeInfo>& type_info,
    bool traverse_tests = false);

}  // namespace xls::dslx

#endif  // XLS_DSLX_CPP_EXTRACT_CONVERSION_ORDER_H_
