// Copyright 2020 The XLS Authors
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

#ifndef XLS_DSLX_INTERP_BINDINGS_H_
#define XLS_DSLX_INTERP_BINDINGS_H_

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"

namespace xls::dslx {

// Notes symbolic bindings that correspond to a module name / function name.
struct FnCtx {
  std::string module_name;
  std::string fn_name;
  ParametricEnv parametric_env;

  std::string ToString() const {
    return absl::StrFormat(
        "FnCtx{module_name=\"%s\", fn_name=\"%s\", parametric_env=%s}",
        module_name, fn_name, parametric_env.ToString());
  }
};

// Represents the set of bindings (ident: value mappings) for evaluation.
//
//   Acts as a {ident: Value} mapping that can easily "chain" onto an existing
//   set of bindings when you enter a new binding scope; e.g. new bindings may
//   be created in a loop body that you want to discard when you proceed past
//   the loop body.
class InterpBindings {
 public:
  using Entry =
      std::variant<InterpValue, TypeAlias*, EnumDef*, StructDef*, Module*>;

  static std::string_view VariantAsString(const Entry& e);

  explicit InterpBindings(const InterpBindings* parent = nullptr);

  // Various forms of binding additions (identifier to value / AST node).

  void AddValue(std::string identifier, InterpValue value) {
    map_.insert_or_assign(std::move(identifier), Entry(std::move(value)));
  }
  void AddModule(std::string identifier, Module* value) {
    map_.insert_or_assign(std::move(identifier), Entry(value));
  }

  // Resolution functions from identifiers to values / AST nodes.

  absl::StatusOr<InterpValue> ResolveValueFromIdentifier(
      std::string_view identifier, const Span* ref_span,
      const FileTable& file_table) const;

  absl::StatusOr<Module*> ResolveModule(std::string_view identifier) const;

  // Resolves an entry for "identifier" via local mapping and transitive binding
  // parents. Returns nullopt if it is not found.
  std::optional<Entry> ResolveEntry(std::string_view identifier) const;

  // Returns all the keys in this bindings object and all transitive parents as
  // a set.
  absl::flat_hash_set<std::string> GetKeys() const;

  const std::optional<FnCtx>& fn_ctx() const { return fn_ctx_; }

 private:
  // Bindings from the outer scope, may be nullptr.
  const InterpBindings* parent_;

  // Maps an identifier to its bound entry.
  absl::flat_hash_map<std::string, Entry> map_;

  // The current (module name, function name, symbolic bindings) that these
  // Bindings are being used with.
  std::optional<FnCtx> fn_ctx_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_INTERP_BINDINGS_H_
