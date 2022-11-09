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

#ifndef XLS_DSLX_TYPE_INFO_H_
#define XLS_DSLX_TYPE_INFO_H_

#include "xls/dslx/ast.h"
#include "xls/dslx/concrete_type.h"
#include "xls/dslx/symbolic_bindings.h"

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
  absl::flat_hash_map<SymbolicBindings, StartAndWidth> bindings_to_start_width;
};

// Parametric instantiation information related to an invocation AST node.
struct InvocationData {
  // Invocation/Spawn AST node.
  const Invocation* node;
  // Map from symbolic bindings in the caller to the corresponding symbolic
  // bindings in the callee for this invocation.
  absl::flat_hash_map<SymbolicBindings, SymbolicBindings> symbolic_bindings_map;
  // Type information that is specialized for a particular parametric
  // instantiation of an invocation.
  absl::flat_hash_map<SymbolicBindings, TypeInfo*> instantiations;

  std::string ToString() const;
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
  // Type information can be "differential"; e.g. when we obtain type
  // information for a particular parametric instantiation the type information
  // is backed by the enclosing type information for the module. Therefore, type
  // information objects can have a "parent" they delegate queries to if they
  // can't satisfy the information from their local mappings.
  TypeInfo* parent() const { return parent_; }

  // Notes start/width for a slice operation found during type inference.
  void AddSliceStartAndWidth(Slice* node,
                             const SymbolicBindings& symbolic_bindings,
                             StartAndWidth start_width);

  // Retrieves the start/width pair for a given slice, see comment on SliceData.
  std::optional<StartAndWidth> GetSliceStartAndWidth(
      Slice* node, const SymbolicBindings& symbolic_bindings) const;

  // Notes caller/callee relation of symbolic bindings at an invocation.
  //
  // This is kept from type inferencing time for convenience purposes (so it
  // doesn't need to be recalculated anywhere; e.g. in the interpreter).
  //
  // Args:
  //   invocation: The invocation node that (may have) caused parametric
  //     instantiation.
  //   caller: The caller's symbolic bindings at the point of invocation.
  //   callee: The callee's computed symbolic bindings for the invocation.
  void AddInvocationCallBindings(const Invocation* call,
                                    SymbolicBindings caller,
                                    SymbolicBindings callee);

  // Adds derived type info for a parametric invocation.
  //
  // Parametric invocations have /derived/ type information, where the
  // parametric expressions are concretized, and have concrete types
  // corresponding to AST nodes in the instantiated parametric function.
  //
  // Args:
  //   invocation: The invocation the type information has been generated for.
  //   caller: The caller's symbolic bindings that caused this instantiation to
  //     occur.
  //   type_info: The type information that has been determined for this
  //     instantiation.
  //
  // Note that the type_info may be nullptr in special cases, like when mapping
  // a callee which is not parametric.
  void SetInvocationTypeInfo(const Invocation* invocation,
                             SymbolicBindings caller, TypeInfo* type_info);

  // Attempts to retrieve "instantiation" type information -- that is, when
  // there's an invocation with parametrics in a caller, it may map to
  // particular type-information for the callee.
  std::optional<TypeInfo*> GetInvocationTypeInfo(
      const Invocation* invocation, const SymbolicBindings& caller) const;
  absl::StatusOr<TypeInfo*> GetInvocationTypeInfoOrError(
      const Invocation* invocation, const SymbolicBindings& caller) const;

  // Sets the type info for the given proc when typechecked at top-level (i.e.,
  // not via an instantiation). Can only be called on the module root TypeInfo.
  absl::Status SetTopLevelProcTypeInfo(const Proc* p, TypeInfo* ti);

  // Gets the TypeInfo for the given function or proc. Can only [successfully]
  // called on the module root TypeInfo.
  absl::StatusOr<TypeInfo*> GetTopLevelProcTypeInfo(const Proc* p);

  // Sets the type associated with the given AST node.
  void SetItem(const AstNode* key, const ConcreteType& value) {
    XLS_CHECK_EQ(key->owner(), module_);
    dict_[key] = value.CloneToUnique();
  }

  // Attempts to resolve AST node 'key' in the node-to-type dictionary.
  std::optional<ConcreteType*> GetItem(const AstNode* key) const;
  absl::StatusOr<ConcreteType*> GetItemOrError(const AstNode* key) const;

  // Attempts to resolve AST node 'key' to a type with subtype T; e.g:
  //
  //    absl::StatusOr<FunctionType*> f_type =
  //        type_info.GetItemAs<FunctionType>(my_func);
  //
  // If the value is not present, or it is not of the expected type, returns an
  // error status.
  template <typename T>
  absl::StatusOr<T*> GetItemAs(const AstNode* key) const;

  bool Contains(AstNode* key) const;

  // Import AST node based information.
  //
  // Note that added type information and such will generally be owned by the
  // import cache.
  void AddImport(Import* import, Module* module, TypeInfo* type_info);
  std::optional<const ImportedInfo*> GetImported(Import* import) const;
  absl::StatusOr<const ImportedInfo*> GetImportedOrError(Import* import) const;
  const absl::flat_hash_map<Import*, ImportedInfo>& imports() const {
    return imports_;
  }

  // Returns the type information for m, if it is available either as this
  // module or an import of this module.
  std::optional<TypeInfo*> GetImportedTypeInfo(Module* m);

  // Returns whether function "f" requires an implicit token parameter; i.e. it
  // contains a `fail!()` or `cover!()` as determined during type inferencing.
  std::optional<bool> GetRequiresImplicitToken(const Function* f) const;
  void NoteRequiresImplicitToken(const Function* f, bool is_required);

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
  std::optional<const SymbolicBindings*> GetInvocationCalleeBindings(
      const Invocation* invocation, const SymbolicBindings& caller) const;

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
  bool IsKnownConstExpr(const AstNode* node);
  bool IsKnownNonConstExpr(const AstNode* node);
  absl::StatusOr<InterpValue> GetConstExpr(const AstNode* const_expr) const;

  // Retrieves a string that shows the module associated with this type info and
  // which imported modules are present, suitable for debugging.
  std::string GetImportsDebugString() const;

  const absl::flat_hash_map<const Invocation*, InvocationData>&
      invocations() const {
    return invocations_;
  }

  // Returns a reference to the underlying mapping that associates an AST node
  // with its deduced type.
  const absl::flat_hash_map<const AstNode*, std::unique_ptr<ConcreteType>>&
  dict() const {
    return dict_;
  }

 private:
  friend class TypeInfoOwner;

  // Args:
  //  module: The module that owns the AST nodes referenced in the (member)
  //    maps.
  //  parent: Type information that should be queried from the same scope (i.e.
  //    if an AST node is not resolved in the local member maps, the lookup is
  //    then performed in the parent, and so on transitively).
  TypeInfo(Module* module, TypeInfo* parent = nullptr);

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

  Module* module_;
  absl::flat_hash_map<const AstNode*, std::unique_ptr<ConcreteType>> dict_;
  absl::flat_hash_map<Import*, ImportedInfo> imports_;
  absl::flat_hash_map<const Invocation*, InvocationData> invocations_;
  absl::flat_hash_map<Slice*, SliceData> slices_;
  absl::flat_hash_map<const AstNode*, std::optional<InterpValue>> const_exprs_;
  absl::flat_hash_map<const Function*, bool> requires_implicit_token_;

  // Maps a Proc to the TypeInfo used for its top-level typechecking.
  absl::flat_hash_map<const Proc*, TypeInfo*> top_level_proc_type_info_;
  TypeInfo* parent_;  // Note: may be nullptr.
};

// -- Inlines

template <typename T>
inline absl::StatusOr<T*> TypeInfo::GetItemAs(const AstNode* key) const {
  std::optional<ConcreteType*> t = GetItem(key);
  if (!t.has_value()) {
    return absl::NotFoundError(
        absl::StrFormat("No type found for AST node: %s @ %s", key->ToString(),
                        SpanToString(key->GetSpan())));
  }
  XLS_DCHECK(t.value() != nullptr);
  auto* target = dynamic_cast<T*>(t.value());
  if (target == nullptr) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "AST node (%s) @ %s did not have expected ConcreteType subtype.",
        key->GetNodeTypeName(), SpanToString(key->GetSpan())));
  }
  return target;
}

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_INFO_H_
