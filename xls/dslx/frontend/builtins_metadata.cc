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
          {"assert_eq", {.requires_implicit_token = true}},
          {"assert_lt", {.requires_implicit_token = true}},
          {"cover!", {.requires_implicit_token = true}},
          {"fail!", {.requires_implicit_token = true}},
          {"assert!", {.requires_implicit_token = true}},
          {"trace!", {.requires_implicit_token = true}},

          // -- Functions that are represented as AST nodes.
          //
          // Typically built-ins are done as AST nodes if they require some
          // special syntactic construct that is not capable of being handled
          // with the normal grammar.
          {"all_ones!",
           {.is_ast_node = true, .allows_explicit_parametrics = true}},
          {"zero!", {.is_ast_node = true, .allows_explicit_parametrics = true}},
          {"trace_fmt!",
           {.is_ast_node = true, .allows_explicit_parametrics = false}},
          {"assert_fmt!",
           {.is_ast_node = true,
            .requires_implicit_token = true,
            .allows_explicit_parametrics = false}},
          {"vtrace_fmt!",
           {.is_ast_node = true, .allows_explicit_parametrics = false}},

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
          {"bit_slice_update", {}},
          {"ceillog2", {}},
          {"clz", {}},
          {"configured_value_or", {}},
          {"ctz", {}},
          {"gate!", {}},
          {"map", {}},
          {"decode", {}},
          {"encode", {}},
          {"one_hot", {}},
          {"one_hot_sel", {}},
          {"priority_sel", {}},
          {"rev", {}},
          {"umulp", {}},
          // Note: the result tuple from `smulp` are two "bags of bits" that
          // must be added together in order to arrive at the signed product. So
          // we give them back as unsigned and users should cast the sum of
          // these elements to a signed number.
          {"smulp", {}},
          {"array_rev", {}},
          {"array_size", {}},
          {"bit_count", {}},
          {"element_count", {}},
          // Bitwise reduction ops.
          {"and_reduce", {}},
          {"or_reduce", {}},
          {"xor_reduce", {}},
          // Use a dummy value to determine size.
          {"signex", {}},
          {"array_slice", {}},
          {"update", {}},
          {"widening_cast", {}},
          {"checked_cast", {}},

          // Require-const-argument.
          {"range", {}},
          {"zip", {}},

          // -- Proc-oriented built-ins.
          // send/recv (communication) builtins that can only be used within
          // proc scope.
          {"send", {}},
          {"send_if", {}},
          {"recv", {}},
          {"recv_if", {}},
          // non-blocking variants
          {"recv_non_blocking", {}},
          {"recv_if_non_blocking", {}},
          {"join", {}},
          {"token", {}},
          {"read", {}},
          {"write", {}},
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
