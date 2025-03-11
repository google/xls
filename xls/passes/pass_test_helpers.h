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

#ifndef XLS_PASSES_PASS_TEST_HELPERS_H_
#define XLS_PASSES_PASS_TEST_HELPERS_H_

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
namespace xls {

template <typename Inner>
  requires(std::is_base_of_v<OptimizationPass, Inner>)
class RecordIfPassChanged : public OptimizationPass {
 public:
  template <typename... Args>
  RecordIfPassChanged(absl::Nonnull<bool*> changed, Args... args)
      : OptimizationPass("temp_sort_name", "temp_long_name"),
        changed_(changed),
        inner_(std::forward<Args>(args)...) {
    short_name_ = absl::StrCat(inner_.short_name(), "_observer");
    long_name_ = absl::StrCat(inner_.long_name(), " Result Observer");
  }
  bool IsCompound() const override { return inner_.IsCompound(); }
  absl::StatusOr<bool> Run(Package* ir, const OptimizationPassOptions& options,
                           PassResults* results,
                           OptimizationContext& context) const override {
    XLS_ASSIGN_OR_RETURN(*changed_, inner_.Run(ir, options, results, context));
    return *changed_;
  }

 protected:
  absl::StatusOr<bool> RunInternal(
      Package* ir, const OptimizationPassOptions& options, PassResults* results,
      OptimizationContext& context) const override {
    XLS_ASSIGN_OR_RETURN(*changed_, inner_.Run(ir, options, results, context));
    return *changed_;
  }

 private:
  bool* changed_;
  Inner inner_;
};

}  // namespace xls

#endif  // XLS_PASSES_PASS_TEST_HELPERS_H_
