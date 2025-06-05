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

#include "xls/tools/codegen_flags.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/flags/flag.h"
#include "absl/flags/marshalling.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/xls_ir_interface.pb.h"
#include "xls/tools/codegen_flags.pb.h"

// LINT.IfChange
ABSL_FLAG(
    std::string, output_verilog_path, "",
    "Specific output path for the Verilog generated. If not specified then "
    "Verilog is written to stdout.");
ABSL_FLAG(std::string, output_schedule_path, "",
          "Specific output path for the generated pipeline schedule. "
          "If not specified, then no schedule is output.");
ABSL_FLAG(std::string, output_schedule_ir_path, "",
          "Path to write the scheduled IR.");
ABSL_FLAG(std::string, output_block_ir_path, "",
          "Path to write the block-level IR.");
ABSL_FLAG(
    std::string, output_signature_path, "",
    "Specific output path for the module signature. If not specified then "
    "no module signature is generated.");
ABSL_FLAG(std::string, output_verilog_line_map_path, "",
          "Specific output path for Verilog line map. If not specified then "
          "Verilog line map is not generated.");
ABSL_FLAG(std::string, top, "",
          "Top entity of the package to generate the (System)Verilog code.");
ABSL_FLAG(std::string, generator, "pipeline",
          "The generator to use when emitting the device function. Valid "
          "values: pipeline, combinational.");
ABSL_FLAG(
    std::string, input_valid_signal, "",
    "If specified, the emitted module will use an external \"valid\" signal "
    "as the load enable for pipeline registers. The flag value is the "
    "name of the input port for this signal.");
ABSL_FLAG(
    std::string, output_valid_signal, "",
    "The name of the output port which holds the pipelined valid signal.");
ABSL_FLAG(
    std::string, manual_load_enable_signal, "",
    "If specified the load-enable of the pipeline registers of each stage is "
    "controlled via an input port of the indicated name. The width of the "
    "input port is equal to the number of pipeline stages. Bit N of the port "
    "is the load-enable signal for the pipeline registers after stage N.");
ABSL_FLAG(bool, flop_inputs, true,
          "If true, inputs of the module are flopped into registers before "
          "use in generated pipelines. Only used with pipeline generator.");
ABSL_FLAG(bool, flop_outputs, true,
          "If true, the module outputs are flopped into registers before "
          "leaving module. Only used with pipeline generator.");
ABSL_FLAG(std::string, flop_inputs_kind, "flop",
          "Kind of input register to add.  "
          "Valid values: flop, skid, zerolatency.");
ABSL_FLAG(std::string, flop_outputs_kind, "flop",
          "Kind of output register to add.  "
          "Valid values: flop, skid, zerolatency.");
ABSL_FLAG(bool, flop_single_value_channels, true,
          "If false, flop_inputs() and flop_outputs() will not flop "
          "single value channels.");
ABSL_FLAG(bool, add_idle_output, false,
          "If true, an additional idle signal tied to valids of input and "
          "flops is added to the block. This output signal is not registered, "
          "regardless of the setting of flop_outputs. "
          "use in generated pipelines. Only used with pipeline generator.");
ABSL_FLAG(std::string, module_name, "",
          "Explicit name to use for the generated module; if not provided the "
          "mangled IR function name is used.");
ABSL_FLAG(std::string, output_port_name, "out",
          "Explicit name to use for the output port; if not provided the name "
          "\"out\" will be used (only applies to functions).");
ABSL_FLAG(std::string, reset, "",
          "Name of the reset signal. If empty, no reset signal is used.");
ABSL_FLAG(bool, reset_active_low, false,
          "Whether the reset signal is active low.");
ABSL_FLAG(bool, reset_asynchronous, false,
          "Whether the reset signal is asynchronous.");
ABSL_FLAG(bool, reset_data_path, true, "Whether to also reset the datapath.");
ABSL_FLAG(bool, use_system_verilog, true,
          "If true, emit SystemVerilog otherwise emit Verilog.");
ABSL_FLAG(bool, separate_lines, false,
          "If true, emit every subexpression on a separate line.");
ABSL_FLAG(int64_t, max_inline_depth, 5,
          "The maximum depth of subexpressions to include inline. Deeper "
          "expressions are emitted with extra wires/lines to reduce depth.");
ABSL_FLAG(std::string, gate_format, "", "Format string to use for gate! ops.");
ABSL_FLAG(std::string, assert_format, "",
          "Format string to use for assertions.");
ABSL_FLAG(std::string, smulp_format, "", "Format string to use for smulp.");
ABSL_FLAG(std::string, streaming_channel_data_suffix, "",
          "Suffix to append to data signals for streaming channels.");
ABSL_FLAG(std::string, streaming_channel_valid_suffix, "_vld",
          "Suffix to append to valid signals for streaming channels.");
ABSL_FLAG(std::string, streaming_channel_ready_suffix, "_rdy",
          "Suffix to append to ready signals for streaming channels.");
ABSL_FLAG(std::string, umulp_format, "", "Format string to use for smulp.");
ABSL_FLAG(std::vector<std::string>, ram_configurations, {},
          "A comma-separated list of ram configurations, each of the form "
          "ram_name:ram_kind[:<kind-specific configuration>]. For example, a "
          "single-port RAM is specified "
          "ram_name:1RW:request_channel:response_channel:write_comp_name[:"
          "latency] (if "
          "unspecified, latency is assumed to be 1. For a 1RW RAM, the "
          "request channel must be a 4-tuple with entries (addr, wr_data, we, "
          "re) and the response channel must be a 1-tuple with entry "
          "(rd_data). This flag will codegen these channels specially to "
          "support interacting with external RAMs; in particular, the request "
          "data port will expanded into 4 ports for each element of the tuple "
          "(addr, wr_data, we, re). Furthermore, ready/valid ports will be "
          "replaced by internal signals to/from a skid buffer added to catch "
          "the output of the RAM. Note: this flag should generally be used in "
          "conjunction with a scheduling constraint to ensure that the receive "
          "on the response channel comes a cycle after the send on the request "
          "channel.");
ABSL_FLAG(bool, gate_recvs, true,
          "If true, emit logic to gate the data value to zero for a receive "
          "operation in Verilog. Otherwise, the data value is not gated.");
ABSL_FLAG(std::string, fifo_module, "xls_fifo_wrapper",
          "If provided, instantiates the provided module where (positive "
          "width) FIFOs are needed, passing the parameters Width, Depth, "
          "RegisterPushOutputs, RegisterPopOutputs, and EnableBypass; "
          "otherwise, materializes the FIFOs using an internal implementation."
          "See documentation for the requirements this module must meet.");
ABSL_FLAG(std::string, nodata_fifo_module, "",
          "If provided, instantiates the provided module where a no-data FIFO "
          "is required, passing the parameters Depth, RegisterPushOutputs, "
          "RegisterPopOutputs, and EnableBypass; otherwise, materializes the "
          "FIFO using an internal implementation. See documentation for the "
          "requirements this module must meet.");
ABSL_FLAG(bool, materialize_internal_fifos, false,
          "If true and no `--fifo_module` is provided, FIFOs will be "
          "materialized using an internal implementation.");
ABSL_FLAG(bool, array_index_bounds_checking, true,
          "If true, emit bounds checking on array-index operations in Verilog. "
          "Otherwise, the bounds checking is not evaluated.");
ABSL_FLAG(int64_t, max_trace_verbosity, 0,
          "Maximum verbosity for traces. Traces with higher verbosity are "
          "stripped from codegen output. 0 by default.");
ABSL_FLAG(bool, add_invariant_assertions, true,
          "If true, codegen will insert runtime assertions which check that "
          "certain IR-level invariants hold (e.g., one-hot selector "
          "invariants). Disable to omit these assertions.");
ABSL_FLAG(std::string, codegen_options_proto, "",
          "Path to a protobuf containing all codegen args.");
ABSL_FLAG(std::optional<std::string>, codegen_options_used_textproto_file,
          std::nullopt,
          "If present, path to write a protobuf recording all codegen args "
          "used (including those set on the cmd line).");
ABSL_FLAG(std::string, register_merge_strategy, "IdentityOnly",
          "What strategy to use for merging registers. Options are "
          "'IdentityOnly' and, 'DontMerge'/'None'.");
ABSL_FLAG(bool, emit_sv_types, true,
          "Should types annotated with #[sv_type(NAME)] be emitted into "
          "verilog as NAME.");
ABSL_FLAG(std::string, simulation_macro_name, "SIMULATION",
          "Verilog macro name to use in an `ifdef guard for "
          "simulation-specific constructs such as $display statements. If "
          "prefixed with `!` the polarity of the guard is inverted (`ifndef).");
ABSL_FLAG(std::vector<std::string>, assertion_macro_names, {"ASSERT_ON"},
          "Verilog macro names to use in an `ifdef guard for assertions. If "
          "prefixed with `!` the polarity of the guard is inverted (`ifndef).");
ABSL_FLAG(int64_t, codegen_version, 0,
          "Version of codegen to use.  Either 2 (refactored codegen), 1 "
          "(original codegen path), or 0 for default");
ABSL_FLAG(std::string, output_scheduling_pass_metrics_path, "",
          "Output path for the pass pipeline metrics for scheduling passes as "
          "a PassPipelineMetricsProto.");
ABSL_FLAG(std::string, output_codegen_pass_metrics_path, "",
          "Output path for the pass pipeline metrics for codegen  passes as a "
          "PassPipelineMetricsProto.");
ABSL_FLAG(std::string, block_metrics_path, "",
          "The filename to write the metrics, including the bill of "
          "materials, for the generated Verilog file");

struct SeedSeq {
  std::vector<int32_t> elements;
};
ABSL_FLAG(SeedSeq, randomize_order_seed, {},
          "If present, the seed used to randomize the order of lines in the "
          "output. If empty, will use a default order. This can be useful for "
          "creating multiple equivalent Verilog outputs to exercise the rest "
          "of the synthesis pipeline.");

// LINT.ThenChange(
//   //xls/build_rules/xls_providers.bzl,
//   //docs_src/codegen_options.md
// )
// This flag should only be specified by the build-system itself and cannot be
// manually configured.
ABSL_FLAG(
    std::optional<std::string>, ir_interface_proto, std::nullopt,
    "An optional proto which includes details from the original DSLX about the "
    "interface. For example it holds the sv types we want modules to generate");

// Returns a textual flag value corresponding to the SeedSeq `seed_seq`.
std::string AbslUnparseFlag(const SeedSeq& seed_seq) {
  return absl::StrJoin(seed_seq.elements, ",",
                       [](std::string* out, int element) {
                         out->append(absl::UnparseFlag(element));
                       });
}

// Parses a SeedSeq from the command line flag value `text`.
// Returns true and sets `*seed_seq` on success; returns false and sets `*error`
// on failure.
bool AbslParseFlag(std::string_view text, SeedSeq* seed_seq,
                   std::string* error) {
  // We have to clear the list to overwrite any existing value.
  seed_seq->elements.clear();
  // absl::StrSplit("") produces {""}, but we need {} on empty input.
  if (text.empty()) {
    return true;
  }
  for (const auto& part : absl::StrSplit(text, ',')) {
    // Let the flag module parse each element value for us.
    int element;
    if (!absl::ParseFlag(std::string(part), &element, error)) {
      return false;
    }
    seed_seq->elements.push_back(element);
  }
  return true;
}

namespace xls {
namespace {

absl::StatusOr<RegisterMergeStrategyProto> MergeStrategyFromString(
    std::string_view s) {
  if (s == "IdentityOnly" || s == "identity") {
    return STRATEGY_IDENTITY_ONLY;
  }
  if (s == "DontMerge" || s == "None" || s == "none") {
    return STRATEGY_DONT_MERGE;
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Invalid register merge strategy %s. choices: identity, none", s));
}

// Converts flag-provided values for I/O kinds to its proto enum value.
absl::StatusOr<IOKindProto> IOKindProtoFromString(std::string_view s) {
  if (s == "flop") {
    return IO_KIND_FLOP;
  }
  if (s == "skid") {
    return IO_KIND_SKID_BUFFER;
  }
  if (s == "zerolatency") {
    return IO_KIND_ZERO_LATENCY_BUFFER;
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Invalid I/O kind specified: `%s`; choices: flop, skid, zerolatency", s));
}

}  // namespace

static absl::StatusOr<bool> SetOptionsFromFlags(CodegenFlagsProto& proto) {
#define POPULATE_FLAG(__x)                                   \
  {                                                          \
    any_flags_set |= FLAGS_##__x.IsSpecifiedOnCommandLine(); \
    proto.set_##__x(absl::GetFlag(FLAGS_##__x));             \
  }
#define POPULATE_REPEATED_FLAG(__x)                                           \
  {                                                                           \
    any_flags_set |= FLAGS_##__x.IsSpecifiedOnCommandLine();                  \
    do {                                                                      \
      proto.mutable_##__x()->Clear();                                         \
      auto repeated_flag = absl::GetFlag(FLAGS_##__x);                        \
      proto.mutable_##__x()->Add(repeated_flag.begin(), repeated_flag.end()); \
    } while (0);                                                              \
  }
  bool any_flags_set = false;
  POPULATE_FLAG(top);

  // Generator is somewhat special, in that we need to parse it to its enum
  // form.
  std::string generator_str = absl::GetFlag(FLAGS_generator);
  if (generator_str == "pipeline") {
    proto.set_generator(GENERATOR_KIND_PIPELINE);
  } else if (generator_str == "combinational") {
    proto.set_generator(GENERATOR_KIND_COMBINATIONAL);
  } else {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid flag given for -generator; got `%s`", generator_str));
  }

  POPULATE_FLAG(input_valid_signal);
  POPULATE_FLAG(output_valid_signal);
  POPULATE_FLAG(manual_load_enable_signal);
  POPULATE_FLAG(flop_inputs);
  POPULATE_FLAG(flop_outputs);
  POPULATE_FLAG(emit_sv_types);
  POPULATE_FLAG(simulation_macro_name);
  POPULATE_REPEATED_FLAG(assertion_macro_names);

  XLS_ASSIGN_OR_RETURN(
      IOKindProto flop_inputs_kind,
      IOKindProtoFromString(absl::GetFlag(FLAGS_flop_inputs_kind)));
  proto.set_flop_inputs_kind(flop_inputs_kind);
  XLS_ASSIGN_OR_RETURN(
      IOKindProto flop_outputs_kind,
      IOKindProtoFromString(absl::GetFlag(FLAGS_flop_outputs_kind)));
  proto.set_flop_outputs_kind(flop_outputs_kind);

  POPULATE_FLAG(codegen_version);
  POPULATE_FLAG(flop_single_value_channels);
  POPULATE_FLAG(add_idle_output);
  POPULATE_FLAG(module_name);
  POPULATE_FLAG(output_port_name);
  POPULATE_FLAG(reset);
  POPULATE_FLAG(reset_active_low);
  POPULATE_FLAG(reset_asynchronous);
  POPULATE_FLAG(reset_data_path);
  POPULATE_FLAG(use_system_verilog);
  POPULATE_FLAG(separate_lines);
  POPULATE_FLAG(max_inline_depth);
  POPULATE_FLAG(gate_format);
  POPULATE_FLAG(assert_format);
  POPULATE_FLAG(smulp_format);
  POPULATE_FLAG(umulp_format);
  POPULATE_FLAG(streaming_channel_data_suffix);
  POPULATE_FLAG(streaming_channel_valid_suffix);
  POPULATE_FLAG(streaming_channel_ready_suffix);
  POPULATE_REPEATED_FLAG(ram_configurations);
  POPULATE_FLAG(add_invariant_assertions);

  // Optimizations
  POPULATE_FLAG(gate_recvs);
  POPULATE_FLAG(array_index_bounds_checking);
  POPULATE_FLAG(fifo_module);
  POPULATE_FLAG(nodata_fifo_module);
  if (absl::GetFlag(FLAGS_materialize_internal_fifos)) {
    any_flags_set = true;
    if (!FLAGS_fifo_module.IsSpecifiedOnCommandLine()) {
      proto.clear_fifo_module();
    } else {
      XLS_RET_CHECK(proto.fifo_module().empty())
          << "Cannot specify both --fifo_module and "
             "--materialize_internal_fifos.";
      XLS_RET_CHECK(proto.nodata_fifo_module().empty())
          << "Cannot specify both --nodata_fifo_module and "
             "--materialize_internal_fifos.";
    }
  }
  XLS_ASSIGN_OR_RETURN(
      RegisterMergeStrategyProto merge_strategy,
      MergeStrategyFromString(absl::GetFlag(FLAGS_register_merge_strategy)));
  any_flags_set |= FLAGS_register_merge_strategy.IsSpecifiedOnCommandLine();
  proto.set_register_merge_strategy(merge_strategy);

  // Misc
  if (FLAGS_randomize_order_seed.IsSpecifiedOnCommandLine()) {
    any_flags_set = true;
    absl::c_copy(absl::GetFlag(FLAGS_randomize_order_seed).elements,
                 google::protobuf::RepeatedFieldBackInserter(
                     proto.mutable_randomize_order_seed()));
  }
  if (absl::GetFlag(FLAGS_ir_interface_proto)) {
    XLS_ASSIGN_OR_RETURN(
        std::string interface_bytes,
        GetFileContents(*absl::GetFlag(FLAGS_ir_interface_proto)));
    XLS_RET_CHECK(
        proto.mutable_package_interface()->ParseFromString(interface_bytes));
    any_flags_set = true;
    CHECK(proto.has_package_interface());
  }
#undef POPULATE_FLAG
#undef POPULATE_REPEATED_FLAG
  return any_flags_set;
}

absl::StatusOr<CodegenFlagsProto> GetCodegenFlags() {
  CodegenFlagsProto proto;
  XLS_ASSIGN_OR_RETURN(bool any_individual_flags_set,
                       SetOptionsFromFlags(proto));
  if (any_individual_flags_set) {
    if (FLAGS_codegen_options_proto.IsSpecifiedOnCommandLine()) {
      return absl::InvalidArgumentError(
          "Cannot combine 'codegen_options_proto' and codegen arguments");
    }
  } else if (FLAGS_codegen_options_proto.IsSpecifiedOnCommandLine()) {
    XLS_RETURN_IF_ERROR(xls::ParseTextProtoFile(
        absl::GetFlag(FLAGS_codegen_options_proto), &proto));
  }
  if (absl::GetFlag(FLAGS_codegen_options_used_textproto_file)) {
    XLS_RETURN_IF_ERROR(SetTextProtoFile(
        *absl::GetFlag(FLAGS_codegen_options_used_textproto_file), proto));
  }
  return proto;
}

}  // namespace xls
