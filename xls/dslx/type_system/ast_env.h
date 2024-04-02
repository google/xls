// Copyright 2024 The XLS Authors
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

#ifndef XLS_DSLX_TYPE_SYSTEM_AST_ENV_H_
#define XLS_DSLX_TYPE_SYSTEM_AST_ENV_H_

#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {

// Similar to `ParametricEnv` but uses AST nodes as keys.
//
// We create a strong wrapper type around the underlying map (which is not
// recommended style in the general case) because we want to avoid proliferating
// this type and track its creation points -- one cannot convert a ParametricEnv
// into an AstEnv because the AstEnv requires nodes as keys (as they exist in
// the TypeInfo) whereas ParametricEnv only has strings available as keys.
class AstEnv {
 public:
  using KeyT = std::variant<const Param*, const ProcMember*>;
  using MapT = absl::flat_hash_map<KeyT, InterpValue>;

  static NameDef* GetNameDefForKey(KeyT key);

  AstEnv() = default;

  void Add(KeyT key, InterpValue value) {
    CHECK(map_.emplace(key, value).second);
  }

  void Clear() { map_.clear(); }

  MapT::const_iterator begin() const { return map_.begin(); }
  MapT::const_iterator end() const { return map_.end(); }

 private:
  MapT map_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_AST_ENV_H_
