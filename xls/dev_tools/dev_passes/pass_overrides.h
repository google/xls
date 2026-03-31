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

#ifndef XLS_DEV_TOOLS_DEV_PASSES_PASS_OVERRIDES_H_
#define XLS_DEV_TOOLS_DEV_PASSES_PASS_OVERRIDES_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/tools/opt.h"

namespace xls::tools {

// Helper that lets one override passes in the optimization pipeline.
//
// This should be used by passing the result of GetDecorator() to the
// registry_decorator field of OptOptions or calling the result of
// GetDecorator() on the appropriate pass registry.
//
// TODO(allight): We could pretty easily use this to implement most of the
// various opt-flags for controlling the pipeline (eg skip_passes, bisect_limit,
// ir_dump_path, etc.). Its not really clear if doing that sort of refactor is
// really worth it though.
class OptimizationPassOverrides {
 public:
  OptimizationPassOverrides();
  virtual ~OptimizationPassOverrides() = default;
  OptimizationPassOverrides(const OptimizationPassOverrides&) = delete;
  OptimizationPassOverrides& operator=(const OptimizationPassOverrides&) =
      delete;
  OptimizationPassOverrides(OptimizationPassOverrides&&) = delete;
  OptimizationPassOverrides& operator=(OptimizationPassOverrides&&) = delete;

  // Return a pass that should be used in place of the given pass. This could be
  // a wrapper pass that adds functionality or a completely different pass.
  //
  // Generator is used to create instances of the pass that should be replaced.
  //
  // Base registry is the registry that the generator came from. It should be
  // used if one wants to generate other passes without overrides.
  virtual absl::StatusOr<std::unique_ptr<OptimizationPass>> OverridePass(
      const OptimizationPassGenerator& generator,
      const OptimizationPassRegistry& base_registry) = 0;

  // Get a decorator to modify the pass registry to override passes using this
  // class.
  PassRegistryDecorator& decorator() { return *decorator_; };

 private:
  std::unique_ptr<PassRegistryDecorator> decorator_;
};

}  // namespace xls::tools

#endif  // XLS_DEV_TOOLS_DEV_PASSES_PASS_OVERRIDES_H_
