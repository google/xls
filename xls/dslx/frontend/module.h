// Copyright 2023 The XLS Authors
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

#ifndef XLS_DSLX_FRONTEND_MODULE_H_
#define XLS_DSLX_FRONTEND_MODULE_H_

#include <cstdint>
#include <filesystem>  // NOLINT
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/proc.h"

namespace xls::dslx {

using ModuleMember =
    std::variant<Function*, Proc*, TestFunction*, TestProc*, QuickCheck*,
                 TypeAlias*, StructDef*, ProcDef*, ConstantDef*, EnumDef*,
                 Import*, Use*, ConstAssert*, Impl*, VerbatimNode*>;

// Returns all the NameDefs defined by the given module member.
//
// Note that generally there is either zero or one, but for `Use` in particular
// there may be many.
//
// Note: this returns nullptr for constructs that do not define a name, e.g.
// `ConstAssert`.
std::vector<NameDef*> ModuleMemberGetNameDefs(const ModuleMember& mm);

// Returns the starting position of the given module member.
//
// This is sometimes used in reporting; e.g. for test metadata, instead of a
// full span.
Pos GetPos(const ModuleMember& module_member);

std::string_view GetModuleMemberTypeName(const ModuleMember& module_member);

enum class ModuleDirective : uint8_t {
  // Suppresses the "constant naming" warning.
  kAllowNonstandardConstantNaming,

  kAllowNonstandardMemberNaming,

  kTypeInferenceVersion2,

  // Enable "use" syntax instead of "import" for this module.
  kAllowUseSyntax,
};

// Represents a syntactic module in the AST.
//
// Modules contain top-level definitions such as functions and tests.
//
// Attributes:
//   name: Name of this module.
//   top: Top-level module constructs; e.g. functions, tests. Given as a
//    sequence instead of a mapping in case there are unnamed constructs at the
//    module level (e.g. metadata, docstrings).
//   fs_path: Name of the filesystem path that led to this module's AST -- if
//    the AST was constructed in-memory this value will be nullopt. Generally
//    this was relative to the main binary's $CWD (which is often a place like
//    Bazel's execution root) -- this helps output be deterministic even when
//    running distributed compilation.
class Module : public AstNode {
 public:
  Module(std::string name, std::optional<std::filesystem::path> fs_path,
         FileTable& file_table);

  ~Module() override;

  Module(Module&& other) = default;
  Module& operator=(Module&& other) = default;

  AstNodeKind kind() const override { return AstNodeKind::kModule; }

  FileTable& file_name() const {
    CHECK(file_table_ != nullptr);
    return *file_table_;
  }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleModule(this);
  }
  std::optional<Span> GetSpan() const override { return span_; }

  // Note: should generally only be called by the parser once the entire module
  // has been parsed, at which point the extent (limit position) is known.
  //
  // Note: After module parsing, span() is safe to call as an accessor instead
  // of GetSpan().
  void set_span(Span span) { span_ = span; }

  const Span& span() const { return span_.value(); }

  std::string_view GetNodeTypeName() const override { return "Module"; }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  std::string ToString() const override;

  template <typename T, typename... Args>
  T* Make(Args&&... args) {
    static_assert(!std::is_same<T, BuiltinNameDef>::value,
                  "Use Module::GetOrCreateBuiltinNameDef()");
    return MakeInternal<T, Args...>(std::forward<Args>(args)...);
  }

  BuiltinNameDef* GetOrCreateBuiltinNameDef(std::string_view name) {
    auto it = builtin_name_defs_.find(name);
    if (it == builtin_name_defs_.end()) {
      BuiltinNameDef* bnd = MakeInternal<BuiltinNameDef>(std::string(name));
      builtin_name_defs_.emplace_hint(it, std::string(name), bnd);
      return bnd;
    }
    return it->second;
  }

  BuiltinNameDef* GetOrCreateBuiltinNameDef(BuiltinType builtin_type) {
    return GetOrCreateBuiltinNameDef(BuiltinTypeToString(builtin_type));
  }

  using MakeCollisionError = std::function<absl::Status(
      std::string_view module_name, std::string_view member_name,
      const Span& existing_span, const AstNode* existing_node,
      const Span& new_span, const AstNode* new_node)>;

  // Adds a top level "member" to the module. Invokes make_collision_error if
  // there is a naming collision at module scope -- this is done so that errors
  // can be layered appropriately and injected in from outside code (e.g. the
  // parser). If nullptr is given, then a non-positional InvalidArgumentError is
  // raised.
  absl::Status AddTop(ModuleMember member,
                      const MakeCollisionError& make_collision_error);

  // Gets the element in this module with the given target_name, or returns a
  // NotFoundError.
  template <typename T>
  absl::StatusOr<T*> GetMemberOrError(std::string_view target_name) {
    for (ModuleMember& member : top_) {
      if (std::holds_alternative<T*>(member)) {
        T* t = std::get<T*>(member);
        if (t->identifier() == target_name) {
          return t;
        }
      }
    }

    return absl::NotFoundError(
        absl::StrFormat("No %s in module %s with name \"%s\"",
                        T::GetDebugTypeName(), name_, target_name));
  }

  std::optional<Function*> GetFunction(std::string_view target_name);

  // Gets a test construct in this module with the given "target_name", or
  // returns a NotFoundError.
  absl::StatusOr<TestFunction*> GetTest(std::string_view target_name);
  absl::StatusOr<TestProc*> GetTestProc(std::string_view target_name);

  absl::Span<ModuleMember const> top() const { return top_; }
  absl::Span<ModuleMember> top() { return absl::MakeSpan(top_); }

  // Finds the first top-level member in top() with the given "target" name as
  // an identifier.
  std::optional<ModuleMember*> FindMemberWithName(std::string_view target);

  // Returns whether the given node is a public member of this module.
  bool IsPublicMember(const AstNode& node) const;

  const StructDef* FindStructDef(const Span& span) const;

  const ProcDef* FindProcDef(const Span& span) const;

  const EnumDef* FindEnumDef(const Span& span) const;

  // Obtains all the type definition nodes in the module; e.g. TypeAlias,
  // StructDef, EnumDef.
  absl::flat_hash_map<std::string, TypeDefinition> GetTypeDefinitionByName()
      const;

  // Obtains all the type definition nodes in the module in module-member order.
  std::vector<TypeDefinition> GetTypeDefinitions() const;

  absl::StatusOr<TypeDefinition> GetTypeDefinition(std::string_view name) const;

  // Retrieves a constant node from this module with the target name as its
  // identifier, or a NotFound error if none can be found.
  absl::StatusOr<ConstantDef*> GetConstantDef(std::string_view target);

  absl::flat_hash_map<std::string, Import*> GetImportByName() const {
    return GetTopWithTByName<Import>();
  }
  absl::flat_hash_map<std::string, Function*> GetFunctionByName() const {
    return GetTopWithTByName<Function>();
  }
  absl::flat_hash_map<std::string, QuickCheck*> GetQuickCheckByName() const {
    return GetTopWithTByName<QuickCheck>();
  }

  std::vector<QuickCheck*> GetQuickChecks() const {
    return GetTopWithT<QuickCheck>();
  }
  std::vector<StructDef*> GetStructDefs() const {
    return GetTopWithT<StructDef>();
  }
  std::vector<ProcDef*> GetProcDefs() { return GetTopWithT<ProcDef>(); }

  std::vector<Proc*> GetProcs() const { return GetTopWithT<Proc>(); }

  std::vector<Impl*> GetImpls() const { return GetTopWithT<Impl>(); }

  // Returns the identifiers for all functions within this module (in the order
  // in which they are defined).
  std::vector<std::string> GetFunctionNames() const;

  // Returns the identifiers for all tests within this module (in the order in
  // which they are defined).
  std::vector<std::string> GetTestNames() const;

  // Returns the identifiers for all QuickChecks within this module (in the
  // order in which they are defined).
  std::vector<std::string> GetQuickCheckNames() const;

  const std::string& name() const { return name_; }
  const std::optional<std::filesystem::path>& fs_path() const {
    return fs_path_;
  }

  // Finds a node with the given kind and /exactly/ the same span as "target".
  const AstNode* FindNode(AstNodeKind kind, const Span& target) const;

  // Finds all the AST nodes in the module with spans that intercept the given
  // "target" position.
  std::vector<const AstNode*> FindIntercepting(const Pos& target) const;

  // Finds all the AST nodes that fall within the given span.
  std::vector<const AstNode*> FindContained(const Span& target) const;

  // Tags this module as having the given module-level directive "directive".
  void AddDirective(ModuleDirective directive) {
    directives_.insert(directive);
  }

  // Returns all the module-level directive tags.
  const absl::btree_set<ModuleDirective>& directives() const {
    return directives_;
  }

  FileTable* file_table() const { return file_table_; }

 private:
  template <typename T, typename... Args>
  T* MakeInternal(Args&&... args) {
    std::unique_ptr<T> node =
        std::make_unique<T>(this, std::forward<Args>(args)...);
    T* ptr = node.get();
    ptr->SetParentage();
    nodes_.push_back(std::move(node));
    return ptr;
  }

  // Returns all of the elements of top_ that have the given variant type T.
  template <typename T>
  std::vector<T*> GetTopWithT() const {
    std::vector<T*> result;
    for (auto& member : top_) {
      if (std::holds_alternative<T*>(member)) {
        result.push_back(std::get<T*>(member));
      }
    }
    return result;
  }

  // Returns all the elements of top_ that have the given variant type T, using
  // T's identifier as a key. (T must have a string identifier.)
  template <typename T>
  absl::flat_hash_map<std::string, T*> GetTopWithTByName() const {
    absl::flat_hash_map<std::string, T*> result;
    for (auto& member : top_) {
      if (std::holds_alternative<T*>(member)) {
        auto* c = std::get<T*>(member);
        result.insert({c->identifier(), c});
      }
    }
    return result;
  }

  std::string name_;  // Name of this module.

  // Optional filesystem path (may not be present e.g. for DSLX files created in
  // memory).
  std::optional<std::filesystem::path> fs_path_;

  FileTable* file_table_;

  std::vector<ModuleMember> top_;  // Top-level members of this module.
  std::vector<std::unique_ptr<AstNode>> nodes_;  // Lifetime-owned AST nodes.

  // Map of top-level module member name to the member itself.
  absl::flat_hash_map<std::string, ModuleMember> top_by_name_;

  // Builtin name definitions, which we common out on a per-module basis. Not
  // for any particular purpose at this time aside from cleanliness of not
  // having many definition nodes of the same builtin thing floating around.
  absl::flat_hash_map<std::string, BuiltinNameDef*> builtin_name_defs_;

  absl::btree_set<ModuleDirective> directives_;

  // The span of the module is only known once parsing has completed.
  //
  // Implementation note: this could be avoided by using a module builder until
  // all values are completed/availble, but there is a lot of state to carry in
  // the builder and marginal benefit to the builder over the
  // partially-completed object.)
  std::optional<Span> span_;
};

// Helper for making a ternary expression conditional. This avoids the user
// needing to hand-craft the block nodes and such.
Conditional* MakeTernary(Module* module, const Span& span, Expr* test,
                         Expr* consequent, Expr* alternate);

// Returns whether the given module member is annotated as public.
bool IsPublic(const ModuleMember& member);

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_MODULE_H_
