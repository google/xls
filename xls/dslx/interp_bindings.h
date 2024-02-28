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
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type_info.h"

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

  // Creates a new bindings object parented to "parent" and with the additional
  // binding given by name_def_tree/value.
  static InterpBindings CloneWith(InterpBindings* parent,
                                  NameDefTree* name_def_tree,
                                  InterpValue value);

  static std::string_view VariantAsString(const Entry& e);

  explicit InterpBindings(const InterpBindings* parent = nullptr);

  // Various forms of binding additions (identifier to value / AST node).

  // Adds a (tuple) tree of values to the current bindings via name_def_tree.
  //
  // If the frontend types are checked properly, you can zip together the
  // structure of name_def_tree and value. This also handles the case where the
  // name_def_tree is simply a leaf without any tupling (a simple NameDef).
  //
  // Args:
  //   name_def_tree: Tree of identifiers to bind.
  //   value: Value that should have identical structure to name_def_tree (e.g.
  //     if the name_def_tree is (a, b, c) this should be a three-value tuple.
  void AddValueTree(NameDefTree* name_def_tree, InterpValue value);

  void AddValue(std::string identifier, InterpValue value) {
    map_.insert_or_assign(std::move(identifier), Entry(std::move(value)));
  }
  void AddFn(std::string identifier, InterpValue value) {
    CHECK(value.IsFunction());
    map_.insert_or_assign(std::move(identifier), Entry(std::move(value)));
  }
  void AddModule(std::string identifier, Module* value) {
    map_.insert_or_assign(std::move(identifier), Entry(value));
  }
  void AddTypeAlias(std::string identifier, TypeAlias* value) {
    map_.insert_or_assign(std::move(identifier), Entry(value));
  }
  void AddEnumDef(std::string identifier, EnumDef* value) {
    map_.insert_or_assign(std::move(identifier), Entry(value));
  }
  void AddStructDef(std::string identifier, StructDef* value) {
    map_.insert_or_assign(std::move(identifier), Entry(value));
  }
  void AddEntry(std::string identifier, Entry entry) {
    map_.insert_or_assign(std::move(identifier), entry);
  }

  // Resolution functions from identifiers to values / AST nodes.

  absl::StatusOr<InterpValue> ResolveValueFromIdentifier(
      std::string_view identifier, const Span* ref_span = nullptr) const;

  // Resolves a name reference to an interpreter value.
  absl::StatusOr<InterpValue> ResolveValue(const NameRef* name_ref) const {
    return ResolveValueFromIdentifier(name_ref->identifier(),
                                      &name_ref->span());
  }

  absl::StatusOr<Module*> ResolveModule(std::string_view identifier) const;

  // Resolve identifier to a type binding, or returns a status error if it is
  // not found / not a type binding.
  absl::StatusOr<TypeAnnotation*> ResolveTypeAnnotation(
      std::string_view identifier) const;

  absl::StatusOr<std::variant<TypeAnnotation*, EnumDef*, StructDef*>>
  ResolveTypeDefinition(std::string_view identifier) const;

  // Resolves an entry for "identifier" via local mapping and transitive binding
  // parents. Returns nullopt if it is not found.
  std::optional<Entry> ResolveEntry(std::string_view identifier) const;

  // Returns all the keys in this bindings object and all transitive parents as
  // a set.
  absl::flat_hash_set<std::string> GetKeys() const;

  bool Contains(std::string_view key) const {
    return map_.contains(key) || (parent_ != nullptr && parent_->Contains(key));
  }

  void set_fn_ctx(std::optional<FnCtx> value) { fn_ctx_ = std::move(value); }
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
