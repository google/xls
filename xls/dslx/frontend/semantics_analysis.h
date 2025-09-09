// Copyright 2025 The XLS Authors
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

#ifndef XLS_DSLX_FRONTEND_SEMANTICS_ANALYSIS_H_
#define XLS_DSLX_FRONTEND_SEMANTICS_ANALYSIS_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

// This class traverses the AST to generate warnings for improper XLS code that
// is misleading or has no effect. Several kinds of warnings are context free
// and does not require type-checking so they can be emitted at
// RunPreTypeCheckPass, while the others require type-checking and are emitted
// at a later stage. This class also collects and holds necessary information to
// assist the generation of those warnings.
class SemanticsAnalysis {
 public:
  absl::Status RunPreTypeCheckPass(Module& module,
                                   WarningCollector& warning_collector);

  absl::Status RunPostTypeCheckPass(WarningCollector& warning_collector);

  void SetNameDefType(const NameDef* def, const Type* type);

 private:
  // Used by kUnusedDefinition. We cannot completely determine whether a
  // definition is truly unused at RunPreTypeCheckPass, because (1) tokens are
  // implicitly joined, and (2) if we implement const_if, unused definitions in
  // a never-taken branch should not be warned. They have to be reported after
  // type checking.
  std::vector<std::pair<const Function*, std::vector<const NameDef*>>>
      maybe_unreferenced_defs;
  absl::flat_hash_map<const NameDef*, std::unique_ptr<Type>> def_to_type_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_SEMANTICS_ANALYSIS_H_
