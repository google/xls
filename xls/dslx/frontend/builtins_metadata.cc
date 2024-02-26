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

#include "xls/dslx/frontend/builtins_metadata.h"

#include <string>
#include <string_view>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"

namespace xls::dslx {

const absl::flat_hash_map<std::string, BuiltinsData>& GetParametricBuiltins() {
  static const absl::NoDestructor<
      absl::flat_hash_map<std::string, BuiltinsData>>
      map({
          {"add_with_carry", {"(uN[T], uN[T]) -> (u1, uN[T])", false}},
          {"assert_eq", {"(T, T) -> ()", false}},
          {"assert_lt", {"(xN[N], xN[N]) -> ()", false}},
          {"bit_slice", {"(uN[N], uN[U], uN[V]) -> uN[V]", false}},
          {"bit_slice_update", {"(uN[N], uN[U], uN[V]) -> uN[N]", false}},
          {"clz", {"(uN[N]) -> uN[N]", false}},
          {"ctz", {"(uN[N]) -> uN[N]", false}},
          {"cover!", {"(u8[N], u1) -> ()", false}},
          {"fail!", {"(u8[N], T) -> T", false}},
          {"gate!", {"(u1, T) -> T", false}},
          {"map", {"(T[N], (T) -> U) -> U[N]", false}},
          {"decode", {"<uN[M]>(uN[N]) -> uN[M]", false}},
          {"encode", {"(uN[N]) -> uN[ceil(log2(N))]", false}},
          {"one_hot", {"(uN[N], u1) -> uN[N+1]", false}},
          {"one_hot_sel", {"(xN[N], xN[M][N]) -> xN[M]", false}},
          {"priority_sel", {"(xN[N], xN[M][N]) -> xN[M]", false}},
          {"rev", {"(uN[N]) -> uN[N]", false}},
          {"umulp", {"(uN[N], uN[N]) -> (uN[N], uN[N])", false}},

          // Note: the result tuple from `smulp` are two "bags of bits" that
          // must be added together in order to arrive at the signed product. So
          // we give them back as unsigned and users should cast the sum of
          // these elements to a signed number.
          {"smulp", {"(sN[N], sN[N]) -> (uN[N], uN[N])", false}},

          {"array_rev", {"(T[N]) -> T[N]", false}},
          {"array_size", {"(T[N]) -> u32", false}},

          // Bitwise reduction ops.
          {"and_reduce", {"(uN[N]) -> u1", false}},
          {"or_reduce", {"(uN[N]) -> u1", false}},
          {"xor_reduce", {"(uN[N]) -> u1", false}},

          // Use a dummy value to determine size.
          {"signex", {"(xN[M], xN[N]) -> xN[N]", false}},
          {"slice", {"(T[M], uN[N], T[P]) -> T[P]", false}},
          {"trace!", {"(T) -> T", false}},

          // Note: the macros we have AST nodes for.
          //
          // TODO(cdleary): 2023-06-01 I don't remember why, but there was a
          // reason this seemed better than built-ins at the time.
          {"zero!", {.signature = "() -> T", .is_ast_node = true}},
          {"trace_fmt!", {.signature = "(T) -> T", .is_ast_node = true}},

          {"update", {"(T[N], uN[M], T) -> T[N]", false}},
          {"enumerate", {"(T[N]) -> (u32, T)[N]", false}},

          {"widening_cast", {"<U>(T) -> U", false}},
          {"checked_cast", {"<U>(T) -> U", false}},

          // Require-const-argument.
          //
          // Note this is a messed up type signature to need to support and
          // should really be replaced with known-statically-sized iota syntax.
          {"range", {"(const uN[N], const uN[N]) -> uN[N][R]", false}},

          // send/recv (communication) builtins that can only be used within
          // proc scope.
          {"send",
           {.signature = "(token, send_chan<T>, T) -> token",
            .is_ast_node = false}},
          {"send_if",
           {.signature = "(token, send_chan<T>, bool, T) -> token",
            .is_ast_node = false}},

          {"recv",
           {.signature = "(token, recv_chan<T>) -> (token, T)",
            .is_ast_node = false}},
          {"recv_if",
           {.signature = "(token, recv_chan<T>, bool, T) -> (token, T)",
            .is_ast_node = false}},

          // non-blocking variants
          {"recv_non_blocking",
           {.signature = "(token, recv_chan<T>, T) -> (token, T, bool)",
            .is_ast_node = false}},
          {"recv_if_non_blocking",
           {.signature = "(token, recv_chan<T>, bool, T) -> (token, T, bool)",
            .is_ast_node = false}},

          {"join", {.signature = "(token...) -> token", .is_ast_node = false}},
      });

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
