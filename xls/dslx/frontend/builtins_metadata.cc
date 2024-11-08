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
          {"assert_eq", {.signature = "(T, T) -> ()", .is_ast_node = false}},
          {"assert_lt",
           {.signature = "(xN[N], xN[N]) -> ()", .is_ast_node = false}},
          {"bit_slice_update",
           {.signature = "(uN[N], uN[U], uN[V]) -> uN[N]",
            .is_ast_node = false}},
          {"clz", {.signature = "(uN[N]) -> uN[N]", .is_ast_node = false}},
          {"ctz", {.signature = "(uN[N]) -> uN[N]", .is_ast_node = false}},
          {"cover!", {.signature = "(u8[N], u1) -> ()", .is_ast_node = false}},
          {"fail!", {.signature = "(u8[N], T) -> T", .is_ast_node = false}},
          {"assert!",
           {.signature = "(bool, u8[N]) -> ()", .is_ast_node = false}},
          {"gate!", {.signature = "(u1, T) -> T", .is_ast_node = false}},
          {"map",
           {.signature = "(T[N], (T) -> U) -> U[N]", .is_ast_node = false}},
          {"decode",
           {.signature = "<uN[M]>(uN[N]) -> uN[M]", .is_ast_node = false}},
          {"encode",
           {.signature = "(uN[N]) -> uN[ceil(log2(N))]", .is_ast_node = false}},
          {"one_hot",
           {.signature = "(uN[N], u1) -> uN[N+1]", .is_ast_node = false}},
          {"one_hot_sel",
           {.signature = "(uN[N], xN[M][N]) -> xN[M]", .is_ast_node = false}},
          {"priority_sel",
           {.signature = "(uN[N], xN[M][N], xN[M]) -> xN[M]",
            .is_ast_node = false}},
          {"rev", {.signature = "(uN[N]) -> uN[N]", .is_ast_node = false}},
          {"umulp",
           {.signature = "(uN[N], uN[N]) -> (uN[N], uN[N])",
            .is_ast_node = false}},

          // Note: the result tuple from `smulp` are two "bags of bits" that
          // must be added together in order to arrive at the signed product. So
          // we give them back as unsigned and users should cast the sum of
          // these elements to a signed number.
          {"smulp",
           {.signature = "(sN[N], sN[N]) -> (uN[N], uN[N])",
            .is_ast_node = false}},

          {"array_rev", {.signature = "(T[N]) -> T[N]", .is_ast_node = false}},
          {"array_size", {.signature = "(T[N]) -> u32", .is_ast_node = false}},

          // Bitwise reduction ops.
          {"and_reduce", {.signature = "(uN[N]) -> u1", .is_ast_node = false}},
          {"or_reduce", {.signature = "(uN[N]) -> u1", .is_ast_node = false}},
          {"xor_reduce", {.signature = "(uN[N]) -> u1", .is_ast_node = false}},

          // Use a dummy value to determine size.
          {"signex",
           {.signature = "(xN[M], xN[N]) -> xN[N]", .is_ast_node = false}},
          {"slice",
           {.signature = "(T[M], uN[N], T[P]) -> T[P]", .is_ast_node = false}},
          {"trace!", {.signature = "(T) -> T", .is_ast_node = false}},

          // Note: the macros we have AST nodes for.
          //
          // TODO(cdleary): 2023-06-01 I don't remember why, but there was a
          // reason this seemed better than built-ins at the time.
          {"all_ones!", {.signature = "() -> T", .is_ast_node = true}},
          {"zero!", {.signature = "() -> T", .is_ast_node = true}},
          {"trace_fmt!", {.signature = "(T) -> T", .is_ast_node = true}},
          {"vtrace_fmt!", {.signature = "(u32, T) -> T", .is_ast_node = true}},

          {"update",
           {.signature = "(T[...], uN[M]|(uN[M], ...), T) -> T[...]",
            .is_ast_node = false}},
          {"enumerate",
           {.signature = "(T[N]) -> (u32, T)[N]", .is_ast_node = false}},

          {"widening_cast", {.signature = "<U>(T) -> U", .is_ast_node = false}},
          {"checked_cast", {.signature = "<U>(T) -> U", .is_ast_node = false}},

          // Require-const-argument.
          //
          // Note this is a messed up type signature to need to support and
          // should really be replaced with known-statically-sized iota syntax.
          {"range",
           {.signature = "(const uN[N], const uN[N]) -> uN[N][R]",
            .is_ast_node = false}},

          {"zip",
           {.signature = "(T[N], U[N]) -> (T, U)[N]", .is_ast_node = false}},

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
          {"token", {.signature = "() -> token", .is_ast_node = false}},
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
