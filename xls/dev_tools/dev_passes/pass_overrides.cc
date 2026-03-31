// Copyright 2026 The XLS Authors
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

#include "xls/dev_tools/dev_passes/pass_overrides.h"

#include <memory>
#include <string_view>
#include <utility>

#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/tools/opt.h"

namespace xls::tools {

namespace {
class OverridePassGenerator : public OptimizationPassGenerator {
 public:
  OverridePassGenerator(const OptimizationPassRegistryBase& reg,
                        const OptimizationPassRegistry& base,
                        std::unique_ptr<OptimizationPassGenerator> generator,
                        OptimizationPassOverrides* overrides)
      : OptimizationPassGenerator(reg),
        base_(base),
        generator_(std::move(generator)),
        overrides_(overrides) {}

  absl::StatusOr<std::unique_ptr<OptimizationPass>> Generate() const override {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<OptimizationPass> res,
                         overrides_->OverridePass(*generator_, base_));
    return res;
  }

  std::unique_ptr<OptimizationPassGenerator> Clone(
      const OptimizationPassRegistryBase& registry) const override {
    return std::make_unique<OverridePassGenerator>(
        registry, base_, generator_->Clone(registry), overrides_);
  }

 private:
  const OptimizationPassRegistry& base_;
  std::unique_ptr<OptimizationPassGenerator> generator_;
  OptimizationPassOverrides* overrides_;
};

class OverrideDecorator : public PassRegistryDecorator {
 public:
  explicit OverrideDecorator(OptimizationPassOverrides* overrides)
      : overrides_(overrides) {}
  absl::StatusOr<std::unique_ptr<OptimizationPassRegistry>> Decorate(
      const OptimizationPassRegistry& registry) override {
    auto decorated =
        std::make_unique<OptimizationPassRegistry>(registry.OverridableClone());
    for (std::string_view name : registry.GetRegisteredNames()) {
      XLS_ASSIGN_OR_RETURN(auto* generator, registry.Generator(name));
      auto clone = generator->Clone(*decorated);
      XLS_RETURN_IF_ERROR(decorated->Register(
          name, std::make_unique<OverridePassGenerator>(
                    *decorated, registry, std::move(clone), overrides_)));
    }
    return std::move(decorated);
  }

 private:
  OptimizationPassOverrides* overrides_;
};
}  // namespace

OptimizationPassOverrides::OptimizationPassOverrides()
    : decorator_(std::make_unique<OverrideDecorator>(this)) {}

}  // namespace xls::tools
