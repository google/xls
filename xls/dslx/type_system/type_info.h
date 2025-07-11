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

// Support for carrying information from the type inferencing phase.

#ifndef XLS_DSLX_TYPE_SYSTEM_TYPE_INFO_H_
#define XLS_DSLX_TYPE_SYSTEM_TYPE_INFO_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/ret_check.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {

class TypeInfo;

// Information associated with an import node in the AST.
struct ImportedInfo {
  Module* module;
  TypeInfo* type_info;
};

// Represents a (start, width) pair used for a bit-slice operation, as
// determined at type inference time.
struct StartAndWidth {
  int64_t start;
  int64_t width;
};

// Data associated with a slice AST node, associating it with concrete
// start/width values determined at type inferencing time.
struct SliceData {
  Slice* node;
  absl::flat_hash_map<ParametricEnv, StartAndWidth> bindings_to_start_width;
};

// For a given invocation, this is the data we record on the parametric callee
// -- "callee_bindings" notes what the parametric environment is for the callee
// and "derived_type_info" holds the type information that is specific to that
// parametric instantiation.
struct InvocationCalleeData {
  ParametricEnv callee_bindings;
  TypeInfo* derived_type_info;
};

// Parametric instantiation information related to an invocation AST node.
struct InvocationData {
 public:
  InvocationData(const Invocation* node, const Function* callee,
                 const Function* caller,
                 absl::flat_hash_map<ParametricEnv, InvocationCalleeData>
                     env_to_callee_data);

  const Invocation* node() const { return node_; }
  const Function* callee() const { return callee_; }
  const Function* caller() const { return caller_; }

  const absl::flat_hash_map<ParametricEnv, InvocationCalleeData>&
  env_to_callee_data() const {
    return env_to_callee_data_;
  }

  // Adds information for this invocation node when invoked in the environment
  // "caller_env" -- it associates that caller environment with the given
  // "callee_data" (which describes what the callee parametric bindings are and
  // whether there is a derivative type information to start using to resolve
  // types in the callee for that invocation).
  //
  // Returns an error if the `caller_env` is invalid
  absl::Status Add(ParametricEnv caller_env, InvocationCalleeData callee_data);

  std::string ToString() const;

 private:
  // Validates that the keys in "env" are a subset of the parametric keys in
  // "caller()".
  absl::Status ValidateEnvForCaller(const ParametricEnv& env) const;

  // Invocation/Spawn AST node.
  const Invocation* node_;

  const Function* callee_;

  // Function containing the above invocation "node". This is held for
  // "referential integrity" so we can check the validity of the caller
  // environments in "env_to_callee_data".
  //
  // Note that this can be nullptr when the invocation is at the top level, e.g.
  // in a const binding.
  const Function* caller_;

  // Map from symbolic bindings in the caller to the corresponding symbolic
  // bindings in the callee for this invocation.
  absl::flat_hash_map<ParametricEnv, InvocationCalleeData> env_to_callee_data_;
};

// Owns "type information" objects created during the type checking process.
//
// In the process of type checking we may instantiate "sub type-infos" for
// things like particular parametric instantiations, that have
// parametric-independent type information as a parent (see TypeInfo::parent()).
//
// Since we decide to create these "sub type-infos" in a way that is driven by
// the program at type checking time, we place all type info objects into this
// owned pool (arena style ownership to avoid circular references or leaks or
// any other sort of lifetime issues).
class TypeInfoOwner {
 public:
  // Returns an error status iff parent is nullptr and "module" already has a
  // root type info.
  absl::StatusOr<TypeInfo*> New(Module* module, TypeInfo* parent = nullptr);

  // Retrieves the root type information for the given module, or a not-found
  // status error if it is not present.
  absl::StatusOr<TypeInfo*> GetRootTypeInfo(const Module* module);

 private:
  // Mapping from module to the "root" (or "parentmost") type info -- these have
  // nullptr as their parent. There should only be one of these for any given
  // module.
  absl::flat_hash_map<const Module*, TypeInfo*> module_to_root_;

  // Owned type information objects -- TypeInfoOwner is the lifetime owner for
  // these.
  std::vector<std::unique_ptr<TypeInfo>> type_infos_;
};

class TypeInfo {
 public:
  ~TypeInfo();

  // Type information can be "differential"; e.g. when we obtain type
  // information for a particular parametric instantiation the type information
  // is backed by the enclosing type information for the module. Therefore, type
  // information objects can have a "parent" they delegate queries to if they
  // can't satisfy the information from their local mappings.
  TypeInfo* parent() const { return parent_; }

  // Notes start/width for a slice operation found during type inference.
  void AddSliceStartAndWidth(Slice* node, const ParametricEnv& parametric_env,
                             StartAndWidth start_width);

  // Retrieves the start/width pair for a given slice, see comment on SliceData.
  std::optional<StartAndWidth> GetSliceStartAndWidth(
      Slice* node, const ParametricEnv& parametric_env) const;

  // Notes caller/callee relation of parametric env at an invocation.
  //
  // This is kept from type inferencing time for convenience purposes (so it
  // doesn't need to be recalculated anywhere; e.g. in the interpreter).
  //
  // Args:
  //   invocation: The invocation node that (may have) caused parametric
  //     instantiation.
  //   callee: The function being invoked.
  //   caller: The function containing the invocation -- note that this can be
  //     nullptr if the invocation is at the top level of the module.
  //   caller_env: The caller's symbolic bindings at the point of invocation.
  //   callee_env: The callee's computed symbolic bindings for the invocation.
  //
  // Returns an error status if internal invariants are violated; e.g. if the
  // "caller_env" is not a valid env for the "caller".
  absl::Status AddInvocationTypeInfo(const Invocation& invocation,
                                     const Function* callee,
                                     const Function* caller,
                                     const ParametricEnv& caller_env,
                                     const ParametricEnv& callee_env,
                                     TypeInfo* derived_type_info);

  // Add data for a non-parametric invocation.
  absl::Status AddInvocation(const Invocation& invocation,
                             const Function* callee, const Function* caller);

  // Attempts to retrieve "instantiation" type information -- that is, when
  // there's an invocation with parametrics in a caller, it may map to
  // particular type-information for the callee.
  std::optional<TypeInfo*> GetInvocationTypeInfo(
      const Invocation* invocation, const ParametricEnv& caller) const;

  // As above, but returns a NotFound error if the invocation does not have
  // associated type information.
  absl::StatusOr<TypeInfo*> GetInvocationTypeInfoOrError(
      const Invocation* invocation, const ParametricEnv& caller) const;

  // Sets the type info for the given proc when typechecked at top-level (i.e.,
  // not via an instantiation). Can only be called on the module root TypeInfo.
  absl::Status SetTopLevelProcTypeInfo(const Proc* p, TypeInfo* ti);

  // Gets the TypeInfo for the given function or proc. Can only [successfully]
  // called on the module root TypeInfo.
  absl::StatusOr<TypeInfo*> GetTopLevelProcTypeInfo(const Proc* p);

  // Sets the type associated with the given AST node.
  void SetItem(const AstNode* key, const Type& value) {
    CHECK_EQ(key->owner(), module_) << key->ToString();
    dict_[key] = value.CloneToUnique();
  }
  void SetItem(const AstNode* key, std::unique_ptr<Type> value) {
    CHECK_EQ(key->owner(), module_);
    dict_[key] = std::move(value);
  }

  // Attempts to resolve AST node 'key' in the node-to-type dictionary.
  std::optional<Type*> GetItem(const AstNode* key) const;
  absl::StatusOr<Type*> GetItemOrError(const AstNode* key) const;

  // Attempts to resolve AST node 'key' to a type with subtype T; e.g.:
  //
  //    absl::StatusOr<FunctionType*> f_type =
  //        type_info.GetItemAs<FunctionType>(my_func);
  //
  // If the value is not present, or it is not of the expected type, returns an
  // error status.
  template <typename T>
  absl::StatusOr<T*> GetItemAs(const AstNode* key) const;

  bool Contains(AstNode* key) const;

  struct TypeSource {
    TypeInfo* type_info;
    std::variant<StructDef*, ProcDef*, EnumDef*, TypeAlias*> definition;
  };

  // Get the actual instructions which provided the given type-definition with
  // its name.
  absl::StatusOr<TypeSource> ResolveTypeDefinition(TypeDefinition source);
  absl::StatusOr<TypeSource> ResolveTypeDefinition(ColonRef* source);
  absl::StatusOr<TypeSource> ResolveTypeDefinition(UseTreeEntry* source);

  // Find the first annotated sv_type for the given type reference, assuming one
  // exists
  absl::StatusOr<std::optional<std::string>> FindSvType(TypeAnnotation* source);

  // Import AST node based information.
  //
  // Note that added type information and such will generally be owned by the
  // import cache.
  void AddImport(ImportSubject import, Module* module, TypeInfo* type_info);

  // Returns information on the imported module (its module AST node and
  // top-level type information).
  std::optional<const ImportedInfo*> GetImported(Import* import) const;
  std::optional<ImportedInfo*> GetImported(Import* import);

  std::optional<const ImportedInfo*> GetImported(
      UseTreeEntry* use_tree_entry) const;
  std::optional<ImportedInfo*> GetImported(UseTreeEntry* use_tree_entry);

  std::optional<const ImportedInfo*> GetImported(ImportSubject import) const;
  std::optional<ImportedInfo*> GetImported(ImportSubject import);

  absl::StatusOr<const ImportedInfo*> GetImportedOrError(Import* import) const;
  absl::StatusOr<ImportedInfo*> GetImportedOrError(Import* import);

  // Returns the imported module information associated with the given
  // use-tree-entry.
  absl::StatusOr<ImportedInfo*> GetImportedOrError(
      UseTreeEntry* use_tree_entry);
  absl::StatusOr<const ImportedInfo*> GetImportedOrError(
      const UseTreeEntry* use_tree_entry) const;

  // As above but takes the "generic import" variant where it may have resolved
  // from either a `UseTreeEntry` or an `Import`.
  absl::StatusOr<ImportedInfo*> GetImportedOrError(ImportSubject import);
  absl::StatusOr<const ImportedInfo*> GetImportedOrError(
      ImportSubject import) const;

  // Returns the type information for m, if it is available either as this
  // module or an import of this module.
  std::optional<TypeInfo*> GetImportedTypeInfo(Module* m);

  // Returns whether function "f" requires an implicit token parameter; i.e. it
  // contains a `fail!()` or `cover!()` as determined during type inferencing.
  std::optional<bool> GetRequiresImplicitToken(const Function& f) const;
  void NoteRequiresImplicitToken(const Function& f, bool is_required);

  // Attempts to retrieve the callee's parametric values in an "instantiation".
  // That is, in the case of:
  //
  //    fn id<M: u32>(x: bits[M]) -> bits[M] { x }
  //    fn p<N: u32>(x: bits[N]) -> bits[N] { id(x) }
  //
  // The invocation of `id(x)` in caller `p` when N=32 would resolve to a record
  // with M=32.
  //
  // When calling a non-parametric callee, the record will be absent.
  std::optional<const ParametricEnv*> GetInvocationCalleeBindings(
      const Invocation* invocation, const ParametricEnv& caller) const;

  Module* module() const { return module_; }

  // Notes the evaluation of a constexpr to a value, as discovered during type
  // checking. Some constructs *require* constexprs, e.g. slice bounds or
  // for-loop range upper limits.
  //
  // Since TypeInfos exist in a tree to indicate parametric instantiation, the
  // note of constexpr evaluation lives on this TypeInfo specifically (it does
  // not automatically get placed in the root of the tree). This avoids
  // collisions in cases e.g. where you slice `[0:N]` where `N` is a parametric
  // value.
  //
  // Note that these index over AstNodes instead of Exprs so that NameDefs can
  // be used as constexpr keys.
  void NoteConstExpr(const AstNode* const_expr, InterpValue value);
  bool IsKnownConstExpr(const AstNode* node) const;
  bool IsKnownNonConstExpr(const AstNode* node) const;
  absl::StatusOr<InterpValue> GetConstExpr(const AstNode* const_expr) const;
  std::optional<InterpValue> GetConstExprOption(
      const AstNode* const_expr) const;

  // Storage of unrolled loops by parametric env.
  void NoteUnrolledLoop(const UnrollFor* loop, const ParametricEnv& env,
                        Expr* unrolled_expr);
  std::optional<Expr*> GetUnrolledLoop(const UnrollFor* loop,
                                       const ParametricEnv& env) const;

  // Retrieves a string that shows the module associated with this type info and
  // which imported modules are present, suitable for debugging.
  std::string GetImportsDebugString() const;

  // Returns a string with the tree of type information (e.g. with
  // what instantiations are present and what the derivated type info pointers
  // are) suitable for debugging.
  std::string GetTypeInfoTreeString() const;

  // Returns the InvocationData for the given invocation, if present in this
  // TypeInfo's root.
  std::optional<const InvocationData*> GetRootInvocationData(
      const Invocation* invocation) const;

  const absl::flat_hash_map<ImportSubject, ImportedInfo>& GetRootImports()
      const {
    return GetRoot()->imports();
  }

  // Returns a reference to the underlying mapping that associates an AST node
  // with its deduced type.
  const absl::flat_hash_map<const AstNode*, std::unique_ptr<Type>>& dict()
      const {
    return dict_;
  }

  const FileTable& file_table() const;
  FileTable& file_table();

 private:
  friend class TypeInfoOwner;

  const absl::flat_hash_map<const Invocation*, std::unique_ptr<InvocationData>>&
  invocations() const {
    CHECK(IsRoot());
    return invocations_;
  }

  // Args:
  //  module: The module that owns the AST nodes referenced in the (member)
  //    maps.
  //  parent: Type information that should be queried from the same scope (i.e.
  //    if an AST node is not resolved in the local member maps, the lookup is
  //    then performed in the parent, and so on transitively).
  explicit TypeInfo(Module* module, TypeInfo* parent = nullptr);

  // Traverses to the 'root' (AKA 'most parent') TypeInfo. This is a place to
  // stash context-free information (e.g. that is found in a parametric
  // instantiation context, but that we want to be accessible to other
  // parametric instantiations).
  TypeInfo* GetRoot() {
    TypeInfo* t = this;
    while (t->parent_ != nullptr) {
      t = t->parent_;
    }
    return t;
  }
  const TypeInfo* GetRoot() const {
    return const_cast<TypeInfo*>(this)->GetRoot();
  }

  // Returns whether this is the root type information for the module (vs. a
  // derived type info for e.g. a parametric instantiation context).
  bool IsRoot() const { return this == GetRoot(); }

  const absl::flat_hash_map<ImportSubject, ImportedInfo>& imports() const {
    return imports_;
  }

  Module* module_;

  // Node to type mapping -- this is present on "derived" type info (i.e. for
  // instantiated parametric type info) as well as the root type information for
  // a module.
  absl::flat_hash_map<const AstNode*, std::unique_ptr<Type>> dict_;

  // Node to constexpr-value mapping -- this is also present on "derived" type
  // info as constexprs take on different values in different parametric
  // instantiation contexts.
  absl::flat_hash_map<const AstNode*, std::optional<InterpValue>> const_exprs_;

  // Unrolled versions of `unroll_for!` loops.
  absl::flat_hash_map<const UnrollFor*,
                      absl::flat_hash_map<ParametricEnv, Expr*>>
      unrolled_loops_;

  // The following are only present on the root type info.
  absl::flat_hash_map<ImportSubject, ImportedInfo> imports_;
  absl::flat_hash_map<const Invocation*, std::unique_ptr<InvocationData>>
      invocations_;
  absl::flat_hash_map<Slice*, SliceData> slices_;
  absl::flat_hash_map<const Function*, bool> requires_implicit_token_;

  // Maps a Proc to the TypeInfo used for its top-level typechecking.
  absl::flat_hash_map<const Proc*, TypeInfo*> top_level_proc_type_info_;

  TypeInfo* parent_;  // Note: may be nullptr.
};

// -- Inlines

template <typename T>
inline absl::StatusOr<T*> TypeInfo::GetItemAs(const AstNode* key) const {
  static_assert(std::is_base_of<Type, T>::value,
                "T must be a subclass of Type");

  std::optional<Type*> t = GetItem(key);
  if (!t.has_value()) {
    return absl::NotFoundError(
        absl::StrFormat("No type found for AST node: %s @ %s", key->ToString(),
                        SpanToString(key->GetSpan(), file_table())));
  }
  DCHECK(t.value() != nullptr);
  auto* target = dynamic_cast<T*>(t.value());
  XLS_RET_CHECK(target != nullptr) << absl::StreamFormat(
      "AST node `%s` @ %s did not have expected `xls::dslx::Type` subtype; "
      "want: %s got: %s",
      key->ToString(), SpanToString(key->GetSpan(), file_table()),
      T::GetDebugName(), t.value()->GetDebugTypeName());
  return target;
}

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_TYPE_INFO_H_
