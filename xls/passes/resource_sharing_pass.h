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

#ifndef XLS_PASSES_RESOURCE_SHARING_PASS_H_
#define XLS_PASSES_RESOURCE_SHARING_PASS_H_

#include <cstdint>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/critical_path_delay_analysis.h"
#include "xls/passes/node_dependency_analysis.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/visibility_analysis.h"

namespace xls {

class ResourceSharingPass : public OptimizationFunctionBasePass {
 public:
  static constexpr std::string_view kName = "resource_sharing";

  // avoids folds with small area savings on certain ops
  static constexpr double kMinAreaSavings = 15.0;
  static constexpr double kMaxDelaySpread = 160.0;
  static constexpr uint64_t kMaxDelayIncrease = 20;
  static constexpr uint64_t kMaxDelayIncreasePerFold = 4;

  // avoids large visibility expressions that will likely be disqualified later
  static constexpr int64_t kMaxEdgesToHandle = 50;
  static constexpr int64_t kMaxPathCountForEdgeInGeneralVisibilityAnalysis = 64;
  static constexpr int64_t kMaxPathCountForBddEngine = 4096;

  explicit ResourceSharingPass();

  ~ResourceSharingPass() override = default;

  enum class ProfitabilityGuard {
    kInDegree,

    kCliques,

    kRandom,  // This heuristic makes this pass not deterministic when different
              // seeds of the PRVG are used.

    kAlways,  // This heuristic applies resource sharing whenever possible. This
              // is used for testing only.
  };

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext& context) const override;

 private:
  ProfitabilityGuard profitability_guard_;

 public:
  struct Analyses {
    const NodeForwardDependencyAnalysis& nda;
    const BddQueryEngine& bdd_engine;
    const VisibilityAnalysis& visibility;
    const SingleSelectVisibilityAnalysis& single_select_visibility;
    const CriticalPathDelayAnalysis& critical_path_delay;
    AreaEstimator& area_estimator;
  };
};

}  // namespace xls

#endif  // XLS_PASSES_RESOURCE_SHARING_PASS_H_
