# Copyright 2021 The XLS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the providers for the XLS build rules."""

load("@bazel_skylib//lib:dicts.bzl", "dicts")

DslxInfo = provider(
    doc = "A provider containing DSLX file information for the target. The " +
          "provider is primarily created and returned by the " +
          "xls_dslx_library rule.",
    fields = {
        "dslx_placeholder_files": "Depset: A depset containing the DSLX generated " +
                                  "placeholder (.placeholder) files of the xls_dslx_library " +
                                  "target and its xls_dslx_library dependencies. " +
                                  "A DSLX placeholder file is generated when the DSLX " +
                                  "source files of a xls_dslx_library rule is " +
                                  "successfully parsed and type checked. " +
                                  "It is used to create a dependency between " +
                                  "xls_dslx_library targets.",
        "dslx_source_files": "Depset: A depset containing the DSLX source " +
                             "(.x) files of the xls_dslx_library " +
                             "target and its xls_dslx_library dependencies. ",
        "target_dslx_source_files": "List: A list containing the DSLX source " +
                                    "(.x) files of the xls_dslx_library " +
                                    "target.",
    },
)

IrFileInfo = provider(
    doc = "A provider containing a single IR file (and any associated metadata as might be " +
          "required). Anything that generates IR should return exactly one of these.",
    fields = {
        "ir_file": "File: The ir file",
    },
)

ConvIrInfo = provider(
    doc = "A provider containing IR conversion file information for the " +
          "target. It is created and returned by the xls_dslx_ir rule.",
    fields = {
        "original_input_files": "List[File]: The original source files.",
        "ir_interface": "File: The IR interface file containing metadata about shape of the " +
                        "design and sv metadata.",
    },
)

OptIrArgInfo = provider(
    doc = "A provider containing IR optimization file information for the " +
          "target. It is created and returned by the xls_ir_opt_ir rule.",
    fields = {
        "opt_ir_args": "Dictionary: The arguments for the IR optimizer.",
        "unopt_ir": "IrFileInfo: The unoptimized input.",
    },
)

CODEGEN_FIELDS = {
    "input_ir": "IrFileInfo: The ir set for codegen",
    "top": "String: Name of top level block in the IR.",
    "generator": "The generator to use when emitting the device function.",
    "input_valid_signal": "If specified, the emitted module will use an " +
                          'external "valid" signal as the load enable for ' +
                          "pipeline registers.",
    "output_valid_signal": "The name of the output port which holds the " +
                           "pipelined valid signal.",
    "manual_load_enable_signal": "If specified the load-enable of the " +
                                 "pipeline registers of each stage is " +
                                 "controlled via an input port of the " +
                                 "indicated name.",
    "flop_inputs": "If true, inputs of the module are flopped into registers " +
                   "before use in generated pipelines.",
    "flop_inputs_kind": "Kind of input register to add.",
    "flop_outputs": "If true, the module outputs are flopped into registers " +
                    "before leaving module.",
    "flop_outputs_kind": "Kind of output register to add.",
    "flop_single_value_channels": "If false, flop_inputs() and " +
                                  "flop_outputs() will not flop single value " +
                                  "channels.",
    "add_idle_output": "If true, an additional idle signal tied to valids of " +
                       "input and flops is added to the block.",
    "module_name": "Explicit name to use for the generated module; if not " +
                   "provided the mangled IR function name is used.",
    "output_port_name": "Explicit name to use for the output port; if not " +
                        "provided the name \"out\" will be used (only applies to functions).",
    "reset": "Name of the reset signal.",
    "reset_active_low": "Whether the reset signal is active low.",
    "reset_asynchronous": "Whether the reset signal is asynchronous.",
    "reset_data_path": "Whether to also reset the datapath.",
    "use_system_verilog": "If true, emit SystemVerilog otherwise emit Verilog.",
    "separate_lines": "If true, emit every subexpression on a separate line.",
    "max_inline_depth": "The maximum depth of subexpressions to include inline; " +
                        "deeper expressions are emitted with extra wires/lines to reduce depth.",
    "streaming_channel_data_suffix": "Suffix to append to data signals for " +
                                     "streaming channels.",
    "streaming_channel_ready_suffix": "Suffix to append to ready signals for " +
                                      "streaming channels.",
    "streaming_channel_valid_suffix": "Suffix to append to valid signals for " +
                                      "streaming channels.",
    "assert_format": "Format string to use for assertions.",
    "gate_format": "Format string to use for gate! ops.",
    "smulp_format": "Format string to use for smulp.",
    "umulp_format": "Format string to use for smulp.",
    "ram_configurations": "A comma-separated list of ram configurations.",
    "gate_recvs": "If true, emit logic to gate the data value to zero for a " +
                  "receive operation in Verilog.",
    "array_index_bounds_checking": "If true, emit bounds checking on " +
                                   "array-index operations in Verilog.",
    "max_trace_verbosity": "Maximum verbosity for traces. Traces with higher " +
                           "verbosity are stripped from codegen output. 0 by " +
                           "default.",
    "add_invariant_assertions": "If true, codegen will insert runtime assertions " +
                                "which check that  certain IR-level invariants hold " +
                                "(e.g., one-hot selector invariants). Disable to " +
                                "omit these assertions.",
    "register_merge_strategy": "The strategy to use for merging registers. Either " +
                               "'IdentityOnly' or 'None'",
    "emit_sv_types": "Whether or not to honor the #[sv_type(NAME)] annotations in the source DSLX.",
    "codegen_version": "Version of codegen to use (0=default).",
    "fifo_module": "If provided, instantiates the provided module where (positive " +
                   "width) FIFOs are needed, passing the parameters Width, Depth, " +
                   "RegisterPushOutputs, RegisterPopOutputs, and EnableBypass; " +
                   "if passed an empty string, materializes the FIFOs using an internal " +
                   "implementation. If not provided, defaults to `xls_fifo_wrapper` (unless " +
                   "`materialize_internal_fifos` is set). See documentation for the requirements " +
                   "this module must meet.",
    "nodata_fifo_module": "If provided, instantiates the provided module where a no-data FIFO is " +
                          "required, passing the parameters Depth, RegisterPushOutputs, " +
                          "RegisterPopOutputs, and EnableBypass; otherwise, materializes the " +
                          "FIFO using an internal implementation. See documentation for the " +
                          "requirements this module must meet.",
    "materialize_internal_fifos": "If true and `fifo_module` is not provided, materializes FIFOs " +
                                  "using an internal implementation. Will produce an error if " +
                                  "`fifo_module` is provided with a non-empty string.",
    "randomize_order_seed": "If present, the seed used to randomize the order of lines in the " +
                            "output, as a comma-separated list of one or more 32-bit integers. " +
                            "If empty, will use a default order. This can be useful for creating " +
                            "multiple equivalent Verilog outputs to exercise a synthesis pipeline.",
}

SCHEDULING_FIELDS = {
    "opt_level": "Optional(int): The optimization level used during scheduling.",
    "clock_period_ps": "Optional(string): The clock period used for " +
                       "scheduling.",
    "pipeline_stages": "Optional(string): The number of pipeline stages.",
    "delay_model": "Optional(string) Delay model used in codegen.",
    "clock_margin_percent": "The percentage of clock period to set aside as " +
                            "a margin to ensure timing is met.",
    "period_relaxation_percent": "The percentage of clock period that will " +
                                 "be relaxed when scheduling without an " +
                                 "explicit --clock_period_ps.",
    "minimize_clock_on_failure": "If true, when `--clock_period_ps` is given " +
                                 "but is infeasible for scheduling, search for " +
                                 "& report the shortest feasible clock period.",
    "recover_after_minimizing_clock": "If this and `--minimize_clock_on_failure` are both given, " +
                                      "when `--clock_period_ps` is given but is infeasible for " +
                                      "scheduling, will print a warning and continue scheduling " +
                                      "as if the shortest feasible clock period had been given.",
    "minimize_worst_case_throughput": "If true, when `--worst_case_throughput` " +
                                      "is not given, search for & report the best " +
                                      "possible worst-case throughput of the circuit " +
                                      "(subject to all other constraints). If " +
                                      "`--clock_period_ps` is not set, will first " +
                                      "optimize for clock speed, and then find the best " +
                                      "possible worst-case throughput within that constraint.",
    "worst_case_throughput": "Allow scheduling a pipeline with worst-case throughput " +
                             "no slower than once per N cycles. If unspecified and " +
                             "`--minimize_worst_case_throughput` is not set, defaults to 1 " +
                             "(full throughput).\n" +
                             "\n" +
                             "If zero or negative, no throughput bound will be enforced.",
    "dynamic_throughput_objective_weight": "If set, the scheduler will attempt to optimize for " +
                                           "dynamic throughput as well as for area; the value " +
                                           "controls how strongly this is prioritized. e.g., if " +
                                           "set to 1024.0 (the default value), the scheduler " +
                                           "will consider improving the dynamic throughput of " +
                                           "one state element by 1 cycle (assuming that all " +
                                           "data-dependent feedback paths are equally likely) to " +
                                           "be worth adding up to 1024 flops. Only relevant if " +
                                           "using the SDC scheduler with --worst_case_throughput " +
                                           "set to a value != 1.",
    "additional_input_delay_ps": "The additional delay added to each input. Note that " +
                                 "flow-controlled channel operations all have inputs and " +
                                 "outputs, so this delay is added to sends and receives.",
    "additional_output_delay_ps": "The additional delay added to each output. Note that " +
                                  "flow-controlled channel operations all have inputs and " +
                                  "outputs, so this delay is added to sends and receives.",
    "additional_channel_delay_ps": "The additional delay added to each specified external " +
                                   "channel's operations, as a comma-separated list of " +
                                   "channel=delay pairs. Note that flow-controlled channel " +
                                   "operations all have both inputs and outputs, so the overall " +
                                   "delay added is the sum of this and the maximum of the " +
                                   "specified additional_(input|output)_delay_ps parameters " +
                                   "(if provided).",
    "ffi_fallback_delay_ps": "Delay of foreign function calls if not " +
                             "otherwise specified.",
    "io_constraints": "A comma-separated list of IO constraints.",
    "receives_first_sends_last": "If true, this forces receives into the " +
                                 "first cycle and sends into the last cycle.",
    "mutual_exclusion_z3_rlimit": "Resource limit for solver in mutual " +
                                  "exclusion pass.",
    "default_next_value_z3_rlimit": "Resource limit for solver when optimizing " +
                                    "default next_value omission; skipped if " +
                                    "negative or omitted.",
    "explain_infeasibility": "If scheduling fails, re-run scheduling with " +
                             "extra slack variables in an attempt to explain " +
                             "why scheduling failed.",
    "infeasible_per_state_backedge_slack_pool": "If specified, the specified " +
                                                "value must be > 0.",
    "use_fdo": "Optional(bool): Enable FDO when true.",
    "fdo_iteration_number": "Optional(int): How many scheduling " +
                            "iterations to run (minimum 2).",
    "fdo_delay_driven_path_number": "The number of delay-driven subgraphs in " +
                                    "each FDO iteration.",
    "fdo_fanout_driven_path_number": "The number of fanout-driven subgraphs " +
                                     "in each FDO iteration.",
    "fdo_refinement_stochastic_ratio": "Must be a positive float <= 1.0.",
    "fdo_path_evaluate_strategy": "Support window, cone, and path for now.",
    "fdo_synthesizer_name": "Only support yosys for now.",
    "fdo_yosys_path": "Absolute path of Yosys.",
    "fdo_sta_path": "Absolute path of OpenSTA.",
    "fdo_synthesis_libraries": "Synthesis and STA libraries.",
    "fdo_default_driver_cell": "Cell to assume is driving primary inputs.",
    "fdo_default_load": "Cell to assume is being driven by primary outputs.",
    "multi_proc": "If true, schedule all procs and codegen them all.",
    "simulation_macro_name": "Name of the Verilog macro used to guard simulation-only " +
                             "constructs. If prefixed with `!` the polarity of the guard " +
                             "is inverted.",
    "assertion_macro_names": "Names of the Verilog macros used to guard assertions. If prefixed " +
                             "with `!` the polarity of the guard is inverted.",
}

_VERILOG_FIELDS = {
    "schedule_ir_file": "File: The IR file post-scheduling.",
    "block_ir_file": "File: The block IR file.",
    "module_sig_file": "File: The module signature of the Verilog file.",
    "schedule_file": "File: The schedule of the module.",
    "verilog_line_map_file": "File: The Verilog line map file.",
    "verilog_file": "File: The Verilog file.",
    "block_metrics_file": "File: The block metrics file.",
}

_CODEGEN_FIELDS = dicts.add(CODEGEN_FIELDS, SCHEDULING_FIELDS, _VERILOG_FIELDS)
CodegenInfo = provider(
    doc = "A provider containing Codegen file information for the target. It " +
          "is created and returned by the xls_ir_verilog rule.",
    fields = _CODEGEN_FIELDS,
)

JitWrapperInfo = provider(
    doc = "A provider containing JIT Wrapper file information for the " +
          "target. It is created and returned by the xls_ir_jit_wrapper rule.",
    fields = {
        "header_file": "File: The header file.",
        "source_file": "File: The source file.",
    },
)

AotCompileInfo = provider(
    doc = "A provider containing the object code for a compiled function " +
          "and the proto describing the compiled code.",
    fields = {
        "object_file": "File: The object code file",
        "proto_file": "File: The protobuf AotEntrypointProto message.",
    },
)

XlsOptimizationPassInfo = provider(
    doc = "A provider containing the implementation of a optimization-pass and its registration code",
    fields = {
        "pass_impl": "CcInfo: The cc info for the actual pass code",
        "pass_registration": "CcInfo: The cc info for the library which registers the pass with the injection system.",
    },
)

XlsOptimizationPassRegistryInfo = provider(
    doc = "A provider containing a set of passes to add to a registry",
    fields = {
        "passes": "List of CcInfo: The cc info for the library which registers each pass with the injection system.",
        "pipeline_binpb": "File: binary proto of the pass pipeline used for this registry.",
    },
)
