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

bool IsAstNodeBuiltinWithExplicitParametrics(std::string_view identifier) {
  return identifier == "zero!" || identifier == "all_ones!";
}

const absl::flat_hash_map<std::string, BuiltinsData>& GetParametricBuiltins() {
  static const absl::NoDestructor<
      absl::flat_hash_map<std::string, BuiltinsData>>
      map({
          // -- Functions that require implicit tokens
          {"assert_eq",
           {.signature = "(T, T) -> ()", .requires_implicit_token = true}},
          {"assert_lt",
           {.signature = "(xN[N], xN[N]) -> ()",
            .requires_implicit_token = true}},
          {"cover!",
           {.signature = "(u8[N], u1) -> ()", .requires_implicit_token = true}},
          {"fail!",
           {.signature = "(u8[N], T) -> T", .requires_implicit_token = true}},
          {"assert!",
           {.signature = "(bool, u8[N]) -> ()",
            .requires_implicit_token = true}},
          {"trace!",
           {.signature = "(T) -> T", .requires_implicit_token = true}},

          // -- Functions that are represented as AST nodes.
          //
          // Typically built-ins are done as AST nodes if they require some
          // special syntactic construct that is not capable of being handled
          // with the normal grammar.
          {"all_ones!",
           {.signature = "() -> T",
            .is_ast_node = true,
            .requires_implicit_token = false,
            .allows_explicit_parametrics = true}},
          {"zero!",
           {.signature = "() -> T",
            .is_ast_node = true,
            .requires_implicit_token = false,
            .allows_explicit_parametrics = true}},
          {"trace_fmt!",
           {.signature = "(T) -> T",
            .is_ast_node = true,
            .requires_implicit_token = false,
            .allows_explicit_parametrics = false}},
          {"vtrace_fmt!",
           {.signature = "(u32, T) -> T",
            .is_ast_node = true,
            .requires_implicit_token = false,
            .allows_explicit_parametrics = false}},

          // -- Normal built-in functions
          //
          // Functions with a `!` suffix are typically indicating that they have
          // superpowers that cannot be written as a normal user function. This
          // is akin to its use as a "special macro expansion indicator" in
          // Rust.
          //
          // Some builtins, though they could be written as user defined
          // functions, are best kept as specially recognized entities so they
          // can be passed directly as primitives to the XLS IR without
          // decomposing them and losing any associated high-level semantic
          // information in the process.
          {"bit_slice_update", {.signature = "(uN[N], uN[U], uN[V]) -> uN[N]"}},
          {"ceillog2", {.signature = "(uN[N]) -> uN[N]"}},
          {"clz", {.signature = "(uN[N]) -> uN[N]"}},
          {"configured_value_or", {.signature = "(u8[N], T) -> T"}},
          {"ctz", {.signature = "(uN[N]) -> uN[N]"}},
          {"gate!", {.signature = "(u1, T) -> T"}},
          {"map", {.signature = "(T[N], (T) -> U) -> U[N]"}},
          {"decode", {.signature = "<uN[M]>(uN[N]) -> uN[M]"}},
          {"encode", {.signature = "(uN[N]) -> uN[ceil(log2(N))]"}},
          {"one_hot", {.signature = "(uN[N], u1) -> uN[N+1]"}},
          {"one_hot_sel", {.signature = "(uN[N], xN[M][N]) -> xN[M]"}},
          {"priority_sel", {.signature = "(uN[N], xN[M][N], xN[M]) -> xN[M]"}},
          {"rev", {.signature = "(uN[N]) -> uN[N]"}},
          {"umulp", {.signature = "(uN[N], uN[N]) -> (uN[N], uN[N])"}},

          // Note: the result tuple from `smulp` are two "bags of bits" that
          // must be added together in order to arrive at the signed product. So
          // we give them back as unsigned and users should cast the sum of
          // these elements to a signed number.
          {"smulp", {.signature = "(sN[N], sN[N]) -> (uN[N], uN[N])"}},

          {"array_rev", {.signature = "(T[N]) -> T[N]"}},
          {"array_size", {.signature = "(T[N]) -> u32"}},

          {"bit_count", {.signature = "() -> u32"}},
          {"element_count", {.signature = "() -> u32"}},

          // Bitwise reduction ops.
          {"and_reduce", {.signature = "(uN[N]) -> u1"}},
          {"or_reduce", {.signature = "(uN[N]) -> u1"}},
          {"xor_reduce", {.signature = "(uN[N]) -> u1"}},

          // Use a dummy value to determine size.
          {"signex", {.signature = "(xN[M], xN[N]) -> xN[N]"}},
          {"array_slice", {.signature = "(T[M], uN[N], T[P]) -> T[P]"}},

          {"update",
           {.signature = "(T[...], uN[M]|(uN[M], ...), T) -> T[...]"}},
          {"enumerate", {.signature = "(T[N]) -> (u32, T)[N]"}},

          {"widening_cast", {.signature = "<U>(T) -> U"}},
          {"checked_cast", {.signature = "<U>(T) -> U"}},

          // Require-const-argument.
          //
          // Note this is a messed up type signature to need to support and
          // should really be replaced with known-statically-sized iota syntax.
          {"range", {.signature = "(const uN[N], const uN[N]) -> uN[N][R]"}},

          {"zip", {.signature = "(T[N], U[N]) -> (T, U)[N]"}},

          // -- Proc-oriented built-ins.

          // send/recv (communication) builtins that can only be used within
          // proc scope.
          {"send", {.signature = "(token, send_chan<T>, T) -> token"}},
          {"send_if", {.signature = "(token, send_chan<T>, bool, T) -> token"}},

          {"recv", {.signature = "(token, recv_chan<T>) -> (token, T)"}},
          {"recv_if",
           {.signature = "(token, recv_chan<T>, bool, T) -> (token, T)"}},

          // non-blocking variants
          {"recv_non_blocking",
           {.signature = "(token, recv_chan<T>, T) -> (token, T, bool)"}},
          {"recv_if_non_blocking",
           {.signature = "(token, recv_chan<T>, bool, T) -> (token, T, bool)"}},

          {"join", {.signature = "(token...) -> token"}},
          {"token", {.signature = "() -> token"}},
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
