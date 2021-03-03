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

#ifndef XLS_DSLX_IMPORT_ROUTINES_H_
#define XLS_DSLX_IMPORT_ROUTINES_H_

#include "xls/dslx/ast.h"
#include "xls/dslx/interp_bindings.h"
#include "xls/dslx/type_info.h"

namespace xls::dslx {

// An entry that goes into the ImportCache.
struct ModuleInfo {
  std::unique_ptr<Module> module;
  TypeInfo* type_info;
};

// Immutable "tuple" of tokens that name an absolute import location.
//
// e.g. ("std",) or ("xls", "examples", "foo")
//
// Hashable (usable in a flat hash map).
class ImportTokens {
 public:
  static absl::StatusOr<ImportTokens> FromString(absl::string_view module_name);

  explicit ImportTokens(std::vector<std::string> pieces)
      : pieces_(std::move(pieces)) {}

  template <typename H>
  friend H AbslHashValue(H h, const ImportTokens& self) {
    for (const std::string& s : self.pieces_) {
      h = H::combine(std::move(h), s);
    }
    return h;
  }

  bool operator==(const ImportTokens& other) const {
    if (pieces_.size() != other.pieces_.size()) {
      return false;
    }
    for (int64 i = 0; i < pieces_.size(); ++i) {
      if (pieces_[i] != other.pieces_[i]) {
        return false;
      }
    }
    return true;
  }

  // Returns the "dotted" version of the components that would appear in an
  // import statement; e.g. "xls.examples.foo"
  std::string ToString() const { return absl::StrJoin(pieces_, "."); }

  // The underlying components.
  const std::vector<std::string>& pieces() const { return pieces_; }

 private:
  std::vector<std::string> pieces_;
};

// Wrapper around a {subject: module_info} mapping that modules can be imported
// into.
class ImportCache {
 public:
  bool Contains(const ImportTokens& target) const {
    return cache_.find(target) != cache_.end();
  }

  // Note: returned pointer is not stable across mutations.
  absl::StatusOr<const ModuleInfo*> Get(const ImportTokens& subject) const;

  // Note: returned pointer is not stable across mutations.
  absl::StatusOr<const ModuleInfo*> Put(const ImportTokens& subject,
                                        ModuleInfo module_info);

  TypeInfoOwner& type_info_owner() { return type_info_owner_; }

  // Helper that gets the "root" type information for the module of the given
  // node. (Note that type information lives in a tree configuration where
  // parametric specializations live under the root, see TypeInfo.)
  absl::StatusOr<TypeInfo*> GetRootTypeInfoForNode(AstNode* node);
  absl::StatusOr<TypeInfo*> GetRootTypeInfo(Module* module);

  // The "top level bindings" for a given module are the values that get
  // resolved at module scope on import. Keeping these on the ImportCache avoids
  // recomputing them.
  //
  // Note: in the future having the bindings at this scope will also allow us to
  // note work-in-progress status that can be observed in both type inferencing
  // and interpretation.
  absl::optional<InterpBindings*> GetTopLevelBindings(Module* module);

  // Notes the top level bindings object for the given module.
  //
  // Precondition: bindings should not already be set for the given module, or
  // this will check-fail.
  void SetTopLevelBindings(Module* module, std::unique_ptr<InterpBindings> tlb);

 private:
  absl::flat_hash_map<ImportTokens, ModuleInfo> cache_;
  absl::flat_hash_map<Module*, std::unique_ptr<InterpBindings>>
      top_level_bindings_;
  TypeInfoOwner type_info_owner_;
};

// Type-checking callback lambda.
using TypecheckFn = std::function<absl::StatusOr<TypeInfo*>(Module*)>;

// Imports the module identified (globally) by 'subject'.
//
// Importing means: locating, parsing, typechecking, and caching in the import
// cache.
//
// Resolves against an existing import in 'cache' if it is present.
//
// Args:
//  ftypecheck: Function that can be used to get type information for a module.
//  subject: Tokens that globally uniquely identify the module to import; e.g.
//      something built-in like ('std',) for the standard library or something
//      fully qualified like ('xls', 'lib', 'math').
//  additional_search_paths: Paths to search in addition to the default ones.
//  cache: Cache that we resolve against so we don't waste resources
//      re-importing things in the import DAG.
//
// Returns:
//  The imported module information.
absl::StatusOr<const ModuleInfo*> DoImport(
    const TypecheckFn& ftypecheck, const ImportTokens& subject,
    absl::Span<std::string const> additional_search_paths, ImportCache* cache,
    const Span& import_span);

}  // namespace xls::dslx

#endif  // XLS_DSLX_IMPORT_ROUTINES_H_
