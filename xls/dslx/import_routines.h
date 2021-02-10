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
#include "xls/dslx/type_info.h"

namespace xls::dslx {

// An entry that goes into the ImportCache.
struct ModuleInfo {
  std::unique_ptr<Module> module;
  TypeInfoOwner type_info;
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
  ImportCache() { XLS_VLOG(3) << "Creating ImportCache @ " << this; }

  ~ImportCache() {
    XLS_VLOG(3) << "Destroying ImportCache @ " << this;
  }

  bool Contains(const ImportTokens& target) const {
    return cache_.find(target) != cache_.end();
  }

  // Note: returned pointer is not stable across mutations.
  absl::StatusOr<const ModuleInfo*> Get(const ImportTokens& subject) const {
    auto it = cache_.find(subject);
    if (it == cache_.end()) {
      return absl::NotFoundError(
          "Module information was not found for import " + subject.ToString());
    }
    return &it->second;
  }

  // Note: returned pointer is not stable across mutations.
  absl::StatusOr<const ModuleInfo*> Put(const ImportTokens& subject,
                                        ModuleInfo module_info) {
    auto it = cache_.insert({subject, std::move(module_info)});
    if (!it.second) {
      return absl::InvalidArgumentError(
          "Module is already loaded for import of " + subject.ToString());
    }
    return &it.first->second;
  }

 private:
  absl::flat_hash_map<ImportTokens, ModuleInfo> cache_;
};

// Type-checking callback lambda.
using TypecheckFn = std::function<absl::StatusOr<TypeInfoOwner>(Module*)>;

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
    absl::Span<std::string const> additional_search_paths, ImportCache* cache);

}  // namespace xls::dslx

#endif  // XLS_DSLX_IMPORT_ROUTINES_H_
