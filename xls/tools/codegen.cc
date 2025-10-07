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

#include "xls/tools/codegen.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/block_generator.h"
#include "xls/codegen/block_metrics.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_residual_data.pb.h"
#include "xls/codegen/codegen_result.h"
#include "xls/codegen/combinational_generator.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/op_override.h"
#include "xls/codegen/pipeline_generator.h"
#include "xls/codegen/ram_configuration.h"
#include "xls/codegen/unified_generator.h"
#include "xls/codegen/verilog_line_map.pb.h"
#include "xls/codegen/xls_metrics.pb.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/ffi_delay_estimator.h"
#include "xls/fdo/synthesizer.h"
#include "xls/ir/function_base.h"
#include "xls/ir/op.h"
#include "xls/ir/verifier.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/pass_metrics.pb.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/pipeline_schedule.pb.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/scheduling_pass.h"
#include "xls/scheduling/scheduling_pass_pipeline.h"
#include "xls/scheduling/scheduling_result.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/scheduling_options_flags.pb.h"

namespace xls {
namespace {

// Set the top of the package from the given codegen flags. Returns an error if
// the top is not set either by the flags or by the IR initially.
absl::Status MaybeSetTop(Package* p,
                         const CodegenFlagsProto& codegen_flags_proto) {
  if (!codegen_flags_proto.top().empty()) {
    XLS_RETURN_IF_ERROR(p->SetTopByName(codegen_flags_proto.top()));
  }
  XLS_RET_CHECK(p->GetTop().has_value())
      << "Package " << p->name() << " needs a top function/proc.";
  return absl::OkStatus();
}

// Data structure holding all the options and metadata for codegen and
// scheduling. This is complicated because codegen options depend on scheduling
// flags and scheduling options depend on codegen flags, plus multiple delay
// estimators are required which contain references to each other. This data
// structure puts them all in one place so the logic can be reused.
struct CodegenMetadata {
  SchedulingOptions scheduling_options;
  verilog::CodegenOptions codegen_options;
  const DelayEstimator* base_estimator = nullptr;
  std::unique_ptr<FfiDelayEstimator> ffi_estimator;
  std::unique_ptr<FirstMatchDelayEstimator> first_match_estimator;
  DelayEstimator* delay_estimator = nullptr;

  static absl::StatusOr<CodegenMetadata> Create(
      Package* package,
      const SchedulingOptionsFlagsProto& scheduling_options_flags_proto,
      const CodegenFlagsProto& codegen_flags_proto, bool with_delay_model) {
    CodegenMetadata metadata;
    XLS_ASSIGN_OR_RETURN(metadata.codegen_options,
                         CodegenOptionsFromProto(codegen_flags_proto));

    if (codegen_flags_proto.generator() == GENERATOR_KIND_COMBINATIONAL) {
      if (with_delay_model) {
        XLS_ASSIGN_OR_RETURN(
            metadata.delay_estimator,
            SetUpDelayEstimator(scheduling_options_flags_proto));
      }

      metadata.codegen_options.generate_combinational(true);
      return metadata;
    }
    // Note: this should already be validated by CodegenFlagsFromAbslFlags().
    CHECK_EQ(codegen_flags_proto.generator(), GENERATOR_KIND_PIPELINE)
        << "Invalid generator kind: "
        << static_cast<int>(codegen_flags_proto.generator());

    metadata.codegen_options.generate_combinational(false);

    XLS_ASSIGN_OR_RETURN(
        metadata.scheduling_options,
        SetUpSchedulingOptions(scheduling_options_flags_proto, package));

    QCHECK(metadata.scheduling_options.pipeline_stages() != 0 ||
           metadata.scheduling_options.clock_period_ps() != 0)
        << "Must specify --pipeline_stages or --clock_period_ps (or both).";

    // Add IO constraints for RAMs.
    for (const xls::verilog::RamConfiguration& ram_config :
         metadata.codegen_options.ram_configurations()) {
      for (const IOConstraint& ram_constraint :
           GetRamConfigurationIOConstraints(ram_config)) {
        metadata.scheduling_options.add_constraint(ram_constraint);
      }
    }

    XLS_ASSIGN_OR_RETURN(metadata.base_estimator,
                         SetUpDelayEstimator(scheduling_options_flags_proto));
    metadata.ffi_estimator = std::make_unique<FfiDelayEstimator>(
        metadata.scheduling_options.ffi_fallback_delay_ps());
    metadata.first_match_estimator = std::make_unique<FirstMatchDelayEstimator>(
        "combined_estimator",
        std::vector<const DelayEstimator*>(
            {metadata.base_estimator, metadata.ffi_estimator.get()}));
    metadata.delay_estimator = metadata.first_match_estimator.get();

    if (package->GetTop().value()->IsProc()) {
      // Force using non-pretty printed codegen when generating procs.
      // TODO(tedhong): 2021-09-25 - Update pretty-printer to support
      //  blocks with flow control.
      metadata.codegen_options.emit_as_pipeline(false);
    }
    return metadata;
  }
};

absl::StatusOr<SchedulingResult> ScheduleFromMetadata(
    Package* p, const CodegenMetadata& metadata) {
  return Schedule(p, metadata.scheduling_options, metadata.delay_estimator);
}

absl::StatusOr<PackageSchedule> DeterminePipelineSchedules(
    GeneratorKind generator_kind, Package* p,
    const PackageSchedule* package_schedule) {
  if (!p->GetTop().has_value()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Package %s has no top function.", p->name()));
  }
  // Codegen for a single combinational proc or function.
  if (package_schedule->GetSchedules().empty()) {
    XLS_RET_CHECK_EQ(generator_kind, GENERATOR_KIND_COMBINATIONAL);
    XLS_ASSIGN_OR_RETURN(PipelineSchedule schedule,
                         PipelineSchedule::SingleStage(p->GetTop().value()));
    return PackageSchedule(std::move(schedule));
  }

  XLS_RET_CHECK_EQ(generator_kind, GENERATOR_KIND_PIPELINE);
  return *package_schedule;
}

absl::StatusOr<verilog::CodegenResult> CodegenPipeline(
    Package* p, const PackageSchedule& package_schedule,
    const verilog::CodegenOptions& codegen_options,
    const DelayEstimator* delay_estimator) {
  XLS_RETURN_IF_ERROR(VerifyPackage(p, /*codegen=*/true));

  if (package_schedule.GetSchedules().size() == 1) {
    const PipelineSchedule& schedule =
        package_schedule.GetSchedule(*p->GetTop());
    return verilog::ToPipelineModuleText(schedule, *p->GetTop(),
                                         codegen_options, delay_estimator);
  }
  return verilog::ToPipelineModuleText(package_schedule, p, codegen_options,
                                       delay_estimator);
}

absl::StatusOr<verilog::CodegenResult> CodegenCombinational(
    Package* p, const verilog::CodegenOptions& codegen_options,
    const DelayEstimator* delay_estimator) {
  return verilog::GenerateCombinationalModule(*p->GetTop(), codegen_options,
                                              delay_estimator);
}

absl::StatusOr<verilog::CodegenResult> CodegenFromMetadata(
    Package* p, GeneratorKind generator_kind, const CodegenMetadata& metadata,
    const PackageSchedule* package_schedule) {
  if (metadata.codegen_options.codegen_version() ==
          verilog::CodegenOptions::Version::kOneDotZero ||
      metadata.codegen_options.codegen_version() ==
          verilog::CodegenOptions::Version::kDefault) {
    if (generator_kind == GENERATOR_KIND_COMBINATIONAL) {
      return CodegenCombinational(p, metadata.codegen_options,
                                  metadata.delay_estimator);
    }
    XLS_RET_CHECK_EQ(generator_kind, GENERATOR_KIND_PIPELINE);
    XLS_RET_CHECK(package_schedule != nullptr);
    return CodegenPipeline(p, *package_schedule, metadata.codegen_options,
                           metadata.delay_estimator);
  }

  // Codegen 2.0.
  XLS_ASSIGN_OR_RETURN(
      PackageSchedule package_schedule_2,
      DeterminePipelineSchedules(generator_kind, p, package_schedule));

  XLS_RETURN_IF_ERROR(VerifyPackage(p, /*codegen=*/true));

  PackageScheduleProto package_pipeline_schedules_proto =
      package_schedule_2.ToProto(*metadata.delay_estimator);

  return verilog::GenerateModuleText(package_schedule_2, p,
                                     metadata.codegen_options,
                                     metadata.delay_estimator);
}

verilog::CodegenOptions::IOKind ToIOKind(IOKindProto p) {
  switch (p) {
    case IO_KIND_INVALID:
    case IO_KIND_FLOP:
      return verilog::CodegenOptions::IOKind::kFlop;
    case IO_KIND_SKID_BUFFER:
      return verilog::CodegenOptions::IOKind::kSkidBuffer;
    case IO_KIND_ZERO_LATENCY_BUFFER:
      return verilog::CodegenOptions::IOKind::kZeroLatencyBuffer;
    default:
      LOG(FATAL) << "Invalid IOKindProto value: " << static_cast<int>(p);
  }
}

absl::StatusOr<SchedulingResult> RunSchedulingPipeline(
    FunctionBase* main, const SchedulingOptions& scheduling_options,
    const DelayEstimator* delay_estimator,
    synthesis::Synthesizer* synthesizer) {
  PassResults results;
  SchedulingPassOptions sched_options;
  sched_options.scheduling_options = scheduling_options;
  sched_options.delay_estimator = delay_estimator;
  sched_options.synthesizer = synthesizer;
  OptimizationContext optimization_context;
  std::unique_ptr<SchedulingCompoundPass> scheduling_pipeline =
      CreateSchedulingPassPipeline(optimization_context, scheduling_options);
  XLS_RETURN_IF_ERROR(main->package()->SetTop(main));
  auto scheduling_context =
      (scheduling_options.schedule_all_procs())
          ? SchedulingContext::CreateForWholePackage(main->package())
          : SchedulingContext::CreateForSingleFunction(main);
  absl::Status scheduling_status =
      scheduling_pipeline
          ->Run(main->package(), sched_options, &results, scheduling_context)
          .status();
  if (!scheduling_status.ok()) {
    if (absl::IsResourceExhausted(scheduling_status)) {
      // Resource exhausted error indicates that the schedule was
      // infeasible. Emit a meaningful error in this case.
      std::string error_message = "Design cannot be scheduled";
      if (scheduling_options.pipeline_stages().has_value()) {
        absl::StrAppendFormat(&error_message, " in %d stages",
                              scheduling_options.pipeline_stages().value());
      }
      if (scheduling_options.clock_period_ps().has_value()) {
        absl::StrAppendFormat(&error_message, " with a %dps clock",
                              scheduling_options.clock_period_ps().value());
      }
      return xabsl::StatusBuilder(scheduling_status).SetPrepend()
             << error_message << ": ";
    }
    return scheduling_status;
  }
  XLS_RET_CHECK(scheduling_context.package_schedule().HasSchedule(main));
  return SchedulingResult{
      .package_schedule =
          scheduling_context.package_schedule().ToProto(*delay_estimator),
      .pass_pipeline_metrics = results.ToProto()};
}

}  // namespace

absl::StatusOr<verilog::CodegenOptions> CodegenOptionsFromProto(
    const CodegenFlagsProto& p) {
  verilog::CodegenOptions options;

  if (p.generator() == GENERATOR_KIND_PIPELINE) {
    options = verilog::BuildPipelineOptions();

    if (!p.input_valid_signal().empty()) {
      std::optional<std::string_view> output_signal;
      if (!p.output_valid_signal().empty()) {
        output_signal = p.output_valid_signal();
      }
      options.valid_control(p.input_valid_signal(), output_signal);
    } else if (!p.manual_load_enable_signal().empty()) {
      options.manual_control(p.manual_load_enable_signal());
    }
    options.flop_inputs(p.flop_inputs());
    options.flop_outputs(p.flop_outputs());
    options.flop_inputs_kind(ToIOKind(p.flop_inputs_kind()));
    options.flop_outputs_kind(ToIOKind(p.flop_outputs_kind()));

    options.flop_single_value_channels(p.flop_single_value_channels());
    options.add_idle_output(p.add_idle_output());

    // Set invariant assertion emission regardless of generator kind.  Proto3
    // returns the default (false) when the field is unset, so if the field is
    // not present we emit assertions by default via the CodegenOptions default
    // of `true`.
    if (p.has_add_invariant_assertions()) {
      options.add_invariant_assertions(p.add_invariant_assertions());
    }

    if (!p.reset().empty()) {
      options.reset(p.reset(), p.reset_asynchronous(), p.reset_active_low(),
                    p.reset_data_path());
    }
  }

  if (p.has_package_interface()) {
    options.package_interface(p.package_interface());
  }

  if (p.has_reference_residual_data()) {
    options.set_residual_data(p.reference_residual_data());
  }

  if (!p.module_name().empty()) {
    options.module_name(p.module_name());
  }

  if (!p.output_port_name().empty()) {
    options.output_port_name(p.output_port_name());
  }

  // Set invariant assertion emission regardless of generator kind.  Proto3
  // returns the default (false) when the field is unset, so if the field is
  // not present we emit assertions by default via the CodegenOptions default
  // of `true`.
  if (p.has_add_invariant_assertions()) {
    options.add_invariant_assertions(p.add_invariant_assertions());
  }

  options.use_system_verilog(p.use_system_verilog());
  options.separate_lines(p.separate_lines());
  options.max_inline_depth(p.max_inline_depth());

  if (!p.gate_format().empty()) {
    options.SetOpOverride(Op::kGate,
                          verilog::OpOverrideGateAssignment(p.gate_format()));
  }

  if (!p.assert_format().empty()) {
    options.SetOpOverride(Op::kAssert,
                          verilog::OpOverrideAssertion(p.assert_format()));
  }

  if (!p.smulp_format().empty()) {
    options.SetOpOverride(Op::kSMulp,
                          verilog::OpOverrideInstantiation(p.smulp_format()));
  }

  if (!p.umulp_format().empty()) {
    options.SetOpOverride(Op::kUMulp,
                          verilog::OpOverrideInstantiation(p.umulp_format()));
  }

  options.streaming_channel_data_suffix(p.streaming_channel_data_suffix());
  options.streaming_channel_valid_suffix(p.streaming_channel_valid_suffix());
  options.streaming_channel_ready_suffix(p.streaming_channel_ready_suffix());

  if (p.has_emit_sv_types()) {
    options.emit_sv_types(p.emit_sv_types());
  }

  options.set_simulation_macro_name(p.simulation_macro_name());
  options.set_assertion_macro_names(std::vector<std::string>(
      p.assertion_macro_names().begin(), p.assertion_macro_names().end()));

  std::vector<verilog::RamConfiguration> ram_configurations;
  ram_configurations.reserve(p.ram_configurations_size());
  for (const std::string& config_text : p.ram_configurations()) {
    XLS_ASSIGN_OR_RETURN(verilog::RamConfiguration config,
                         verilog::ParseRamConfiguration(config_text));
    ram_configurations.push_back(std::move(config));
  }
  options.ram_configurations(ram_configurations);

  options.gate_recvs(p.gate_recvs());
  options.set_fifo_module(p.fifo_module());
  options.set_nodata_fifo_module(p.nodata_fifo_module());
  options.array_index_bounds_checking(p.array_index_bounds_checking());
  switch (p.register_merge_strategy()) {
    case STRATEGY_DONT_MERGE:
      options.register_merge_strategy(
          verilog::CodegenOptions::RegisterMergeStrategy::kDontMerge);
      break;
    case STRATEGY_IDENTITY_ONLY:
      options.register_merge_strategy(
          verilog::CodegenOptions::RegisterMergeStrategy::kIdentityOnly);
      break;
    case STRATEGY_INVALID:
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unknown merge strategy: %v", p.register_merge_strategy()));
  }

  if (p.has_source_annotation_strategy()) {
    switch (p.source_annotation_strategy()) {
      case ANNOTATION_STRATEGY_NONE:
        options.source_annotation_strategy(
            verilog::CodegenOptions::SourceAnnotationStrategy::kNone);
        break;
      case ANNOTATION_STRATEGY_COMMENT:
        options.source_annotation_strategy(
            verilog::CodegenOptions::SourceAnnotationStrategy::kComment);
        break;
      case ANNOTATION_STRATEGY_DIRECTIVE:
        options.source_annotation_strategy(
            verilog::CodegenOptions::SourceAnnotationStrategy::kLineDirective);
        break;
      default:
        return absl::InvalidArgumentError(
            absl::StrFormat("Unknown source annotation strategy: %v",
                            p.source_annotation_strategy()));
    }
  }

  if (!p.randomize_order_seed().empty()) {
    options.randomize_order_seed(p.randomize_order_seed());
  }

  if (p.has_codegen_version()) {
    options.codegen_version(p.codegen_version());
  }

  return options;
}

absl::StatusOr<SchedulingResult> Schedule(
    Package* p, const SchedulingOptions& scheduling_options,
    const DelayEstimator* delay_estimator) {
  QCHECK(scheduling_options.pipeline_stages() != 0 ||
         scheduling_options.clock_period_ps() != 0)
      << "Must specify --pipeline_stages or --clock_period_ps (or both).";
  synthesis::Synthesizer* synthesizer = nullptr;
  if (scheduling_options.use_fdo() &&
      !scheduling_options.fdo_synthesizer_name().empty()) {
    XLS_ASSIGN_OR_RETURN(synthesizer, SetUpSynthesizer(scheduling_options));
  }
  return RunSchedulingPipeline(*p->GetTop(), scheduling_options,
                               delay_estimator, synthesizer);
}

absl::StatusOr<SchedulingResult> Schedule(
    Package* p,
    const SchedulingOptionsFlagsProto& scheduling_options_flags_proto,
    const CodegenFlagsProto& codegen_flags_proto) {
  XLS_RETURN_IF_ERROR(MaybeSetTop(p, codegen_flags_proto));
  XLS_ASSIGN_OR_RETURN(
      CodegenMetadata metadata,
      CodegenMetadata::Create(p, scheduling_options_flags_proto,
                              codegen_flags_proto, /*with_delay_model=*/true));
  return ScheduleFromMetadata(p, metadata);
}

absl::StatusOr<verilog::CodegenResult> Codegen(
    Package* p,
    const SchedulingOptionsFlagsProto& scheduling_options_flags_proto,
    const CodegenFlagsProto& codegen_flags_proto, bool with_delay_model,
    const PackageSchedule* package_schedule) {
  XLS_RETURN_IF_ERROR(MaybeSetTop(p, codegen_flags_proto));
  XLS_ASSIGN_OR_RETURN(
      CodegenMetadata metadata,
      CodegenMetadata::Create(p, scheduling_options_flags_proto,
                              codegen_flags_proto, with_delay_model));
  return CodegenFromMetadata(p, codegen_flags_proto.generator(), metadata,
                             package_schedule);
}

absl::StatusOr<std::pair<SchedulingResult, verilog::CodegenResult>>
ScheduleAndCodegen(
    Package* p,
    const SchedulingOptionsFlagsProto& scheduling_options_flags_proto,
    const CodegenFlagsProto& codegen_flags_proto, bool with_delay_model) {
  XLS_RETURN_IF_ERROR(MaybeSetTop(p, codegen_flags_proto));
  XLS_ASSIGN_OR_RETURN(
      CodegenMetadata metadata,
      CodegenMetadata::Create(p, scheduling_options_flags_proto,
                              codegen_flags_proto, with_delay_model));

  XLS_RETURN_IF_ERROR(MaybeSetTop(p, codegen_flags_proto));

  SchedulingResult scheduling_result;
  if (codegen_flags_proto.generator() == GENERATOR_KIND_PIPELINE) {
    XLS_ASSIGN_OR_RETURN(scheduling_result, ScheduleFromMetadata(p, metadata));
  }
  XLS_ASSIGN_OR_RETURN(
      PackageSchedule schedules,
      PackageSchedule::FromProto(p, scheduling_result.package_schedule));
  XLS_ASSIGN_OR_RETURN(verilog::CodegenResult codegen_result,
                       CodegenFromMetadata(p, codegen_flags_proto.generator(),
                                           metadata, &schedules));
  return std::make_pair(std::move(scheduling_result),
                        std::move(codegen_result));
}

absl::StatusOr<verilog::CodegenResult> BlockToVerilog(
    Package* p, const CodegenFlagsProto& codegen_flags_proto) {
  XLS_RETURN_IF_ERROR(MaybeSetTop(p, codegen_flags_proto));
  XLS_ASSIGN_OR_RETURN(verilog::CodegenOptions options,
                       CodegenOptionsFromProto(codegen_flags_proto));

  if (!p->GetTop().has_value()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Package %s has no top set.", p->name()));
  }
  FunctionBase* top_fb = p->GetTop().value();
  if (!top_fb->IsBlock()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Top `%s` is not a block.", top_fb->name()));
  }
  Block* top_block = top_fb->AsBlockOrDie();

  verilog::VerilogLineMap verilog_line_map;
  verilog::CodegenResidualData residual_data;
  XLS_ASSIGN_OR_RETURN(
      std::string verilog,
      verilog::GenerateVerilog(top_block, options, &verilog_line_map,
                               &residual_data));

  verilog::ModuleSignature signature;
  if (top_block->GetSignature().has_value()) {
    XLS_ASSIGN_OR_RETURN(signature, verilog::ModuleSignature::FromProto(
                                        *top_block->GetSignature()));
  }

  verilog::XlsMetricsProto metrics;
  XLS_ASSIGN_OR_RETURN(
      *metrics.mutable_block_metrics(),
      verilog::GenerateBlockMetrics(top_block, /*delay_estimator=*/nullptr));

  return verilog::CodegenResult{.verilog_text = std::move(verilog),
                                .verilog_line_map = std::move(verilog_line_map),
                                .signature = std::move(signature),
                                .block_metrics = std::move(metrics),
                                .residual_data = std::move(residual_data)};
}

}  // namespace xls
