// Copyright 2022 The XLS Authors
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
#ifndef XLS_DSLX_TRAIT_VISITOR_H_
#define XLS_DSLX_TRAIT_VISITOR_H_

#include <vector>

#include "absl/status/status.h"
#include "xls/dslx/ast.h"

namespace xls::dslx {

// Collects various traits/etc. about a DSLX Expr tree.
// Lazily populated as information is needed.
class TraitVisitor : public ExprVisitorWithDefault {
 public:
  absl::Status HandleNameRef(const NameRef* expr) override;

  const std::vector<const NameRef*>& name_refs() { return name_refs_; }

 private:
  std::vector<const NameRef*> name_refs_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TRAIT_VISITOR_H_
