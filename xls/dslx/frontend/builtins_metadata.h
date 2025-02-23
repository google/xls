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

#ifndef XLS_DSLX_FRONTEND_BUILTINS_METADATA_H_
#define XLS_DSLX_FRONTEND_BUILTINS_METADATA_H_

#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"

namespace xls::dslx {

struct BuiltinsData {
  std::string signature;

  // Indicates whether this builtin is represented in the AST as a "first class
  // node" or as an invocation of a builtin name-definition. Most builtins don't
  // need to be AST nodes and thus this is false by default, which is the most
  // common case.
  bool is_ast_node = false;

  // Indicates whether this builtin requires an implicit token parameter when it
  // is used/invoked within a function. For most builtins this is not required,
  // but built-ins that demand a token like `assert!`, `cover!`, `fail!` etc do
  // have this set.
  bool requires_implicit_token = false;
};

// Map from the name of the parametric builtin function; e.g. `assert_eq` to a
// struct that shows 1) the parametric signature; for example: `(T, T) -> ()`
// and 2) whether the builtin is represented as an AST node.
const absl::flat_hash_map<std::string, BuiltinsData>& GetParametricBuiltins();

// Returns whether the identifier is a builtin parameetric function (i.e. a key
// in the `GetParametricBuiltins` map)
// -- built-in functions are always available at the DSLX top level scope, but
// are not implemented as AST nodes.
//
// Warning: prefer to use `IsBuiltinParametricNameRef()` over this function
// wherever possible -- it's easy to forget to check that the name definition is
// a `BuiltinNameDef` before testing the identifier.
bool IsNameParametricBuiltin(std::string_view identifier);

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_BUILTINS_METADATA_H_
