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

#include "xls/dslx/builtins_metadata.h"

namespace xls::dslx {

const absl::flat_hash_map<std::string, BuiltinsData>& GetParametricBuiltins() {
  static const auto* map = new absl::flat_hash_map<std::string, BuiltinsData>{
      {"add_with_carry", {"(uN[T], uN[T]) -> (u1, uN[T])", false}},
      {"assert_eq", {"(T, T) -> ()", false}},
      {"assert_lt", {"(T, T) -> ()", false}},
      {"bit_slice", {"(uN[N], uN[U], uN[V]) -> uN[V]", false}},
      {"bit_slice_update", {"(uN[N], uN[U], uN[V]) -> uN[N]", false}},
      {"clz", {"(uN[N]) -> uN[N]", false}},
      {"ctz", {"(uN[N]) -> uN[N]", false}},
      {"cover!", {"(u8[N], u1) -> ()", false}},
      {"fail!", {"(u8[N], T) -> T", false}},
      {"gate!", {"(u1, T) -> T", false}},
      {"map", {"(T[N], (T) -> U) -> U[N]", false}},
      {"one_hot", {"(uN[N], u1) -> uN[N+1]", false}},
      {"one_hot_sel", {"(xN[N], xN[M][N]) -> xN[M]", false}},
      {"priority_sel", {"(xN[N], xN[M][N]) -> xN[M]", false}},
      {"rev", {"(uN[N]) -> uN[N]", false}},
      {"umulp", {"(uN[N], uN[N]) -> (uN[N], uN[N])", false}},
      {"smulp", {"(sN[N], sN[N]) -> (sN[N], sN[N])", false}},

      // Bitwise reduction ops.
      {"and_reduce", {"(uN[N]) -> u1", false}},
      {"or_reduce", {"(uN[N]) -> u1", false}},
      {"xor_reduce", {"(uN[N]) -> u1", false}},

      // Use a dummy value to determine size.
      {"signex", {"(xN[M], xN[N]) -> xN[N]", false}},
      {"slice", {"(T[M], uN[N], T[P]) -> T[P]", false}},
      {"trace!", {"(T) -> T", false}},
      {"trace_fmt!", {"(T) -> T", true}},
      {"update", {"(T[N], uN[M], T) -> T[N]", false}},
      {"enumerate", {"(T[N]) -> (u32, T)[N]", false}},

      // Require-const-argument.
      //
      // Note this is a messed up type signature to need to support and should
      // really be replaced with known-statically-sized iota syntax.
      {"range", {"(const uN[N], const uN[N]) -> uN[N][R]", false}},
  };

  return *map;
}

bool IsNameParametricBuiltin(std::string_view identifier) {
  const absl::flat_hash_map<std::string, BuiltinsData>& parametric_builtins =
      GetParametricBuiltins();
  if (auto it = parametric_builtins.find(identifier);
      it != parametric_builtins.end() && !it->second.is_ast_node) {
    return true;
  }
  return false;
}

}  // namespace xls::dslx
