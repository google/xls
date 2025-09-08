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

#include "absl/status/status.h"
#include "xls/dslx/frontend/ast.h"
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
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_SEMANTICS_ANALYSIS_H_
