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

#include "xls/dslx/concrete_type.h"
#include "xls/dslx/cpp_ast.h"
#include "xls/dslx/symbolic_bindings.h"

namespace xls::dslx {

class TypeInfo;

// Information associated with an import node in the AST.
struct ImportedInfo {
  std::shared_ptr<Module> module;
  std::shared_ptr<TypeInfo> type_info;
};

// Represents a (start, width) pair used for a bit-slice operation, as
// determined at type inference time.
struct StartAndWidth {
  int64 start;
  int64 width;
};

// Data associated with a slice AST node, associating it with concrete
// start/width values determined at type inferencing time.
struct SliceData {
  Slice* node;
  absl::flat_hash_map<SymbolicBindings, StartAndWidth> bindings_to_start_width;

  void Update(const SliceData& other) {
    XLS_CHECK_EQ(node, other.node);
    for (const auto& [bindings, start_width] : other.bindings_to_start_width) {
      bindings_to_start_width[bindings] = start_width;
    }
  }
};

// Parametric instantiation information related to an invocation AST node.
struct InvocationData {
  // Invocation AST node.
  Invocation* node;
  // Map from symbolic bindings in the caller to the corresponding symbolic
  // bindings in the callee for this invocation.
  absl::flat_hash_map<SymbolicBindings, SymbolicBindings> symbolic_bindings_map;
  // Type information that is specialized for a particular parametric
  // instantiation of an invocation.
  absl::flat_hash_map<SymbolicBindings, std::shared_ptr<TypeInfo>>
      instantiations;

  // Adds all of the information in "other" to this InvocationData.
  void Update(const InvocationData& other);

  std::string ToString() const;
};

class TypeInfo {
 public:
  // Args:
  //  module: The module that owns the AST nodes referenced in the (member)
  //    maps.
  //  parent: Type information that should be queried from the same scope (i.e.
  //    if an AST node is not resolved in the local member maps, the lookup is
  //    then performed in the parent, and so on transitively).
  explicit TypeInfo(const std::shared_ptr<Module>& module,
                    std::shared_ptr<TypeInfo> parent = nullptr)
      : module_(module), parent_(parent) {
    XLS_VLOG(3) << "Created type info for module \"" << module_->name()
                << "\" @ " << this << " parent " << parent.get();
  }

  ~TypeInfo() {
    XLS_VLOG(3) << "Destroying type info for module \"" << module_->name()
                << "\" @ " << this << " parent " << parent_.get();
  }

  // Type information can be "differential"; e.g. when we obtain type
  // information for a particular parametric instantiation the type information
  // is backed by the enclosing type information for the module. Therefore, type
  // information objects can have a "parent" they delegate queries to if they
  // can't satisfy the information from their local mappings.
  const std::shared_ptr<TypeInfo>& parent() const { return parent_; }

  // Updates this type information object with data from 'other'.
  void Update(const TypeInfo& other);

  // Returns if there's type info at 'invocation' with given caller bindings.
  bool HasInstantiation(Invocation* invocation,
                        const SymbolicBindings& caller) const;

  // Notes start/width for a slice operation found during type inference.
  void AddSliceStartAndWidth(Slice* node,
                             const SymbolicBindings& symbolic_bindings,
                             StartAndWidth start_width);

  // Retrieves the start/width pair for a given slice, see comment on SliceData.
  absl::optional<StartAndWidth> GetSliceStartAndWidth(
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
  void AddInvocationSymbolicBindings(Invocation* invocation,
                                     SymbolicBindings caller,
                                     SymbolicBindings callee);

  // Adds derived type info for an "instantiation".
  //
  // An "instantiation" is an invocation of a parametric function from some
  // caller context (given by the invocation / caller symbolic bindings). These
  // have /derived/ type information, where the parametric expressions are
  // concretized, and have concrete types corresponding to AST nodes in the
  // instantiated parametric function.
  //
  // Args:
  //   invocation: The invocation the type information has been generated for.
  //   caller: The caller's symbolic bindings that caused this instantiation to
  //     occur.
  //   type_info: The type information that has been determined for this
  //     instantiation.
  void AddInstantiation(Invocation* invocation, SymbolicBindings caller,
                        const std::shared_ptr<TypeInfo>& type_info);

  // Notes a constant definition associated with a given NameDef AST node.
  void NoteConstant(NameDef* name_def, ConstantDef* constant_def) {
    name_to_const_[name_def] = constant_def;
  }

  // Sets the type associated with the given AST node.
  void SetItem(AstNode* key, const ConcreteType& value) {
    dict_[key] = value.CloneToUnique();
  }

  // Attempts to resolve AST node 'key' in the node-to-type dictionary.
  absl::optional<ConcreteType*> GetItem(AstNode* key) const;

  // Attempts to resolve AST node 'key' to a type with subtype T; e.g:
  //
  //    absl::StatusOr<FunctionType*> f_type =
  //        type_info.GetItemAs<FunctionType>(my_func);
  //
  // If the value is not present, or it is not of the expected type, returns an
  // error status.
  template <typename T>
  absl::StatusOr<T*> GetItemAs(AstNode* key) const;

  bool Contains(AstNode* key) const;

  // Import AST node based information.
  void AddImport(Import* import, const std::shared_ptr<Module>& module,
                 const std::shared_ptr<TypeInfo>& type_info);
  absl::optional<const ImportedInfo*> GetImported(Import* import) const;
  const absl::flat_hash_map<Import*, ImportedInfo>& imports() const {
    return imports_;
  }

  // Invocation AST node based information.
  absl::optional<std::shared_ptr<TypeInfo>> GetInstantiation(
      Invocation* invocation, const SymbolicBindings& caller) const;
  absl::optional<const SymbolicBindings*> GetInvocationSymbolicBindings(
      Invocation* invocation, const SymbolicBindings& caller) const;

  const std::shared_ptr<Module>& module() const { return module_; }

  // Returns the text that corresponds to a const int value.
  absl::optional<Expr*> GetConstInt(NameDef* name_def) const;

  // Clears out references to other TypeInfos to avoid circular references
  // causing leaks. Somewhat unfortunately we invoke this from ownership points
  // in Python code to avoid forming cycles, doesn't seem to be an easy way to
  // interop with the Python cycle detector via pybind11.
  //
  // TODO(leary): 2020-10-03 Once this is all ported to C++, we can have type
  // inferencing ownership lifetime.
  void ClearTypeInfoRefsForGc() {
    imports_.clear();
    invocations_.clear();
    parent_ = nullptr;
  }

 private:
  // Traverses to the 'most parent' TypeInfo. This is a place to stash
  // context-free information (e.g. that is found in a parametric instantiation
  // context, but that we want to be accessible to other parametric
  // instantiations).
  TypeInfo* GetTop() {
    TypeInfo* t = this;
    while (t->parent_ != nullptr) {
      t = t->parent_.get();
    }
    return t;
  }
  const TypeInfo* GetTop() const {
    return const_cast<TypeInfo*>(this)->GetTop();
  }

  std::shared_ptr<Module> module_;
  absl::flat_hash_map<AstNode*, std::unique_ptr<ConcreteType>> dict_;
  absl::flat_hash_map<Import*, ImportedInfo> imports_;
  absl::flat_hash_map<NameDef*, ConstantDef*> name_to_const_;
  absl::flat_hash_map<Invocation*, InvocationData> invocations_;
  absl::flat_hash_map<Slice*, SliceData> slices_;
  std::shared_ptr<TypeInfo> parent_;  // Note: may be nullptr.
};

// -- Inlines

template <typename T>
inline absl::StatusOr<T*> TypeInfo::GetItemAs(AstNode* key) const {
  absl::optional<ConcreteType*> t = GetItem(key);
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
