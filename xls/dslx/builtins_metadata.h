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

#ifndef XLS_DSLX_BUILTINS_METADATA_H_
#define XLS_DSLX_BUILTINS_METADATA_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"

namespace xls::dslx {

struct BuiltinsData {
  std::string signature;
  bool is_ast_node;
};

// Map from the name of the parametric builtin function; e.g. `assert_eq` to a
// struct that shows 1) the parametric signature; for example: `(T, T) -> ()`
// and 2) whether the builtin is represented as an AST node.
const absl::flat_hash_map<std::string, BuiltinsData>& GetParametricBuiltins();

// Returns whether the identifier is a builtin parameter not implemented as an
// AST node
bool IsNameParametricBuiltin(std::string_view identifier);

}  // namespace xls::dslx

#endif  // XLS_DSLX_BUILTINS_METADATA_H_
