// Copyright 2021 The XLS Authors
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

// Library that backs the `opt_main` tool's primary functionality.

#ifndef XLS_TOOLS_OPT_H_
#define XLS_TOOLS_OPT_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/macros.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional_ref.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.pb.h"
#include "xls/passes/pass_metrics.pb.h"
#include "xls/passes/pass_pipeline.pb.h"
#include "xls/tools/opt_flags.pb.h"

namespace xls {
class OptimizationPassRegistry;
}  // namespace xls
namespace xls::tools {

class PassRegistryDecorator {
 public:
  virtual ~PassRegistryDecorator() = default;
  // Return a new pass registry based on the one passed in that has been
  // modified as required by this decorator.
  virtual absl::StatusOr<std::unique_ptr<OptimizationPassRegistry>> Decorate(
      const OptimizationPassRegistry& registry) = 0;
};

// Various optimizer options (that generally funnel from the `opt_main` tool to
// this consolidated library).
//
// Deprecated. Use OptFlagsProto as the canonical representation of options.
struct ABSL_DEPRECATED("Use OptFlagsProto to pass configuration information.")
    OptOptions {
  int64_t opt_level = OptFlagsProto::default_instance().opt_level();
  std::string top = std::string(OptFlagsProto::default_instance().top());
  std::string ir_dump_path =
      std::string(OptFlagsProto::default_instance().ir_dump_path());
  std::optional<std::string> ir_path = std::nullopt;
  std::vector<std::string> skip_passes;
  std::optional<int64_t> convert_array_index_to_select = std::nullopt;
  std::optional<int64_t> split_next_value_selects =
      OptFlagsProto::default_instance().split_next_value_selects();
  std::vector<RamRewrite> ram_rewrites = {};
  bool use_context_narrowing_analysis =
      OptFlagsProto::default_instance().use_context_narrowing_analysis();
  bool optimize_for_best_case_throughput =
      OptFlagsProto::default_instance().optimize_for_best_case_throughput();
  bool enable_resource_sharing =
      OptFlagsProto::default_instance().enable_resource_sharing();
  bool force_resource_sharing =
      OptFlagsProto::default_instance().force_resource_sharing();
  std::string area_model =
      std::string(OptFlagsProto::default_instance().area_model());
  // Custom registry to use to get default pipeline and compound passes.
  std::optional<OptimizationPipelineProto> custom_registry = std::nullopt;
  // Decorator to modify the registry before use. Must live for the duration of
  // the optimization.
  absl::optional_ref<PassRegistryDecorator> registry_decorator = std::nullopt;
  std::optional<PassPipelineProto> pass_pipeline = std::nullopt;
  std::optional<int64_t> bisect_limit;
  bool debug_optimizations =
      OptFlagsProto::default_instance().debug_optimizations();
  std::optional<std::string> delay_model = std::nullopt;
};

ABSL_DEPRECATED("Use OptFlags directly instead")
absl::StatusOr<OptFlagsProto> OptFlagsFromOptOptions(const OptOptions& options);

// Metadata which can be optionally returned from
// the optimizer.
struct OptMetadata {
  PassPipelineMetricsProto metrics;
};

// Struct of additional pointers that can be used to modify the way opt runs.
struct OptModifiers {
  absl::optional_ref<PassRegistryDecorator> registry_decorator = std::nullopt;
};

// Helper used in the opt_main tool, optimizes the given IR for a particular
// top-level entity (e.g., function, proc, etc) at the given opt level and
// modifies the package in place.
absl::Status OptimizeIrForTop(Package* package, const OptFlagsProto& options,
                              OptMetadata* metadata = nullptr,
                              const OptModifiers& modifiers = {});

// Helper used in the opt_main tool, optimizes the given IR for a particular
// top-level entity (e.g., function, proc, etc) at the given opt level and
// modifies the package in place.
//
// Deprecated. Use the OptFlagsProto to pass configuration information.
ABSL_DEPRECATE_AND_INLINE()
ABSL_DEPRECATED("Use OptimizeIrForTop with OptFlagsProto instead.")
inline absl::Status OptimizeIrForTop(Package* package,
                                     const OptOptions& options,
                                     OptMetadata* metadata = nullptr) {
  XLS_ASSIGN_OR_RETURN(OptFlagsProto flags, OptFlagsFromOptOptions(options));
  return OptimizeIrForTop(package, flags, metadata,
                          {.registry_decorator = options.registry_decorator});
}

// Helper used in the opt_main tool, optimizes the given IR for a particular
// top-level entity (e.g., function, proc, etc) at the given opt level and
// returns the resulting optimized IR.
absl::StatusOr<std::string> OptimizeIrForTop(
    std::string_view ir, const OptFlagsProto& options,
    OptMetadata* metadata = nullptr, const OptModifiers& modifiers = {});

// Helper used in the opt_main tool, optimizes the given IR for a particular
// top-level entity (e.g., function, proc, etc) at the given opt level and
// returns the resulting optimized IR.
//
// Deprecated. Use the OptFlagsProto to pass configuration information.
ABSL_DEPRECATE_AND_INLINE()
ABSL_DEPRECATED("Use OptimizeIrForTop with OptFlagsProto instead.")
inline absl::StatusOr<std::string> OptimizeIrForTop(
    std::string_view ir, const OptOptions& options,
    OptMetadata* metadata = nullptr) {
  XLS_ASSIGN_OR_RETURN(OptFlagsProto flags, OptFlagsFromOptOptions(options));
  return OptimizeIrForTop(ir, flags, metadata,
                          {.registry_decorator = options.registry_decorator});
}

}  // namespace xls::tools

#endif  // XLS_TOOLS_OPT_H_
