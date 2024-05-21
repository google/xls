# Copyright 2023 The XLS Authors
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

"""
This module contains *FDO-mode* codegen-related build rules for XLS.
"""

load("@rules_hdl//pdk:build_defs.bzl", "StandardCellInfo")
load("@bazel_skylib//lib:dicts.bzl", "dicts")
load(
    "//xls/build_rules:xls_codegen_rules.bzl",
    "validate_verilog_filename",
)
load(
    "//xls/build_rules:xls_common_rules.bzl",
    "append_default_to_args",
    "args_to_string",
    "get_original_input_files_for_xls",
    "get_output_filename_value",
    "get_runfiles_for_xls",
    "get_src_ir_for_xls",
    "get_transitive_built_files_for_xls",
    "is_args_valid",
    "split_filename",
)
load("//xls/build_rules:xls_config_rules.bzl", "CONFIG")
load("//xls/build_rules:xls_ir_rules.bzl", "xls_ir_common_attrs")
load("//xls/build_rules:xls_providers.bzl", "CodegenInfo")
load(
    "//xls/build_rules:xls_toolchains.bzl",
    "xls_toolchain_attrs",
)

_DEFAULT_SCHEDULING_ARGS = {
    "delay_model": "unit",
}

_DEFAULT_CODEGEN_ARGS = {
    "use_fdo": "true",
}

_VERILOG_FILE_EXTENSION = "v"
_SIGNATURE_TEXTPROTO_FILE_EXTENSION = ".sig.textproto"
_SCHEDULE_TEXTPROTO_FILE_EXTENSION = ".schedule.textproto"
_VERILOG_LINE_MAP_TEXTPROTO_FILE_EXTENSION = ".verilog_line_map.textproto"
_BLOCK_IR_FILE_EXTENSION = ".block.ir"
_SCHEDULE_IR_FILE_EXTENSION = ".schedule.opt.ir"

xls_ir_verilog_attrs = {
    "codegen_args": attr.string_dict(
        doc = "Arguments of the codegen tool. For details on the arguments, " +
              "refer to the codegen_main application at " +
              "//xls/tools/codegen_main.cc.",
    ),
    "codegen_options_proto": attr.label(
        allow_single_file = True,
        default = None,
        doc = "Filename of a protobuf with arguments of the codegen tool. " +
              "For details on the arguments, " +
              "refer to the codegen_main application at " +
              "//xls/tools/codegen_main.cc.",
    ),
    "scheduling_options_proto": attr.label(
        allow_single_file = True,
        default = None,
        doc = "Filename of a protobuf with scheduling options arguments " +
              "of the codegen tool. For details on the arguments, " +
              "refer to the codegen_main application at " +
              "//xls/tools/codegen_main.cc.",
    ),
    "verilog_file": attr.output(
        doc = "The filename of Verilog file generated. The filename must " +
              "have a " + _VERILOG_FILE_EXTENSION + " extension.",
        mandatory = True,
    ),
    "module_sig_file": attr.output(
        doc = "The filename of module signature of the generated Verilog " +
              "file. If not specified, the basename of the Verilog file " +
              "followed by a " + _SIGNATURE_TEXTPROTO_FILE_EXTENSION + " " +
              "extension is used.",
    ),
    "schedule_file": attr.output(
        doc = "The filename of schedule of the generated Verilog file." +
              "If not specified, the basename of the Verilog file followed " +
              "by a " + _SCHEDULE_TEXTPROTO_FILE_EXTENSION + " extension is " +
              "used.",
    ),
    "verilog_line_map_file": attr.output(
        doc = "The filename of line map for the generated Verilog file." +
              "If not specified, the basename of the Verilog file followed " +
              "by a " + _VERILOG_LINE_MAP_TEXTPROTO_FILE_EXTENSION + " extension is " +
              "used.",
    ),
    "schedule_ir_file": attr.output(
        doc = "The filename of scheduled IR file generated during scheduled. " +
              "If not specified, the basename of the Verilog file followed " +
              "by a " + _SCHEDULE_IR_FILE_EXTENSION + " extension is " +
              "used.",
    ),
    "block_ir_file": attr.output(
        doc = "The filename of block-level IR file generated during codegen. " +
              "If not specified, the basename of the Verilog file followed " +
              "by a " + _BLOCK_IR_FILE_EXTENSION + " extension is " +
              "used.",
    ),
    "yosys_tool": attr.label(
        default = Label("@at_clifford_yosys//:yosys"),
        executable = True,
        cfg = "exec",
    ),
    "sta_tool": attr.label(
        default = Label("@org_theopenroadproject//:opensta"),
        executable = True,
        cfg = "exec",
    ),
    "standard_cells": attr.label(
        providers = [StandardCellInfo],
        default = "@com_google_skywater_pdk_sky130_fd_sc_hd//:sky130_fd_sc_hd",
    ),
}

def _is_combinational_generator(arguments):
    """Returns True, if "generator" is "combinational". Otherwise, returns False.

    Args:
      arguments: The list of arguments.
    Returns:
      Returns True, if "generator" is "combinational". Otherwise, returns False.
    """
    return arguments.get("generator", "") == "combinational"

def _uses_fdo(arguments):
    """Returns True, if codegen options specify using FDO. Otherwise, returns False.

    Args:
      arguments: The list of arguments.
    Returns:
      Returns True, if codegen will use FDO. Otherwise, returns False.
    """
    return (arguments.get("use_fdo", "false").lower() == "true")

def xls_ir_verilog_fdo_impl(ctx, src, original_input_files):
    """The core implementation of the 'xls_ir_verilog' rule.

    Generates a Verilog file, module signature file, block file, Verilog line
    map, and schedule file.

    Args:
      ctx: The current rule's context object.
      src: The source file.
      original_input_files: All original source files that produced this IR file (used for errors).

    Returns:
      A tuple with the following elements in the order presented:
        1. The CodegenInfo provider
        1. The list of built files.
        1. The runfiles.
    """
    codegen_tool = ctx.executable._xls_codegen_tool
    my_generated_files = []

    # default arguments
    if ctx.file.codegen_options_proto == None:
        codegen_args = append_default_to_args(
            ctx.attr.codegen_args,
            _DEFAULT_CODEGEN_ARGS,
        )
    else:
        codegen_args = ctx.attr.codegen_args

    if ctx.file.scheduling_options_proto == None:
        codegen_args = append_default_to_args(
            codegen_args,
            _DEFAULT_SCHEDULING_ARGS,
        )

    uses_fdo = _uses_fdo(codegen_args)
    if not uses_fdo:
        fail("Cannot set use_fdo to false when using this rule")

    # parse arguments
    CODEGEN_FLAGS = (
        "top",
        "generator",
        "input_valid_signal",
        "output_valid_signal",
        "manual_load_enable_signal",
        "flop_inputs",
        "flop_inputs_kind",
        "flop_outputs",
        "flop_outputs_kind",
        "flop_single_value_channels",
        "add_idle_output",
        "module_name",
        "assert_format",
        "gate_format",
        "reset",
        "reset_active_low",
        "reset_asynchronous",
        "reset_data_path",
        "use_system_verilog",
        "separate_lines",
        "streaming_channel_data_suffix",
        "streaming_channel_ready_suffix",
        "streaming_channel_valid_suffix",
        "assert_format",
        "gate_format",
        "smulp_format",
        "umulp_format",
        "ram_configurations",
        "gate_recvs",
        "array_index_bounds_checking",
        "inline_procs",
    )

    SCHEDULING_FLAGS = (
        "clock_period_ps",
        "pipeline_stages",
        "delay_model",
        "clock_margin_percent",
        "period_relaxation_percent",
        "minimize_clock_on_error",
        "worst_case_throughput",
        "additional_input_delay_ps",
        "ffi_fallback_delay_ps",
        "io_constraints",
        "receives_first_sends_last",
        "mutual_exclusion_z3_rlimit",
        "default_next_value_z3_rlimit",
        "use_fdo",
        "fdo_iteration_number",
        "fdo_delay_driven_path_number",
        "fdo_fanout_driven_path_number",
        "fdo_refinement_stochastic_ratio",
        "fdo_path_evaluate_strategy",
        "fdo_synthesizer_name",
        "fdo_yosys_path",
        "fdo_sta_path",
        "fdo_synthesis_libraries",
    )

    is_args_valid(codegen_args, CODEGEN_FLAGS + SCHEDULING_FLAGS)
    codegen_str_args = args_to_string(codegen_args, CODEGEN_FLAGS + SCHEDULING_FLAGS)
    uses_combinational_generator = _is_combinational_generator(codegen_args)
    final_args = codegen_str_args

    # output filenames
    verilog_filename = ctx.attr.verilog_file.name
    if codegen_args.get("use_system_verilog", "") == "":
        use_system_verilog = split_filename(verilog_filename)[-1] != "v"
    else:
        use_system_verilog = codegen_args.get("use_system_verilog", "True").lower() == "true"
    validate_verilog_filename(verilog_filename, use_system_verilog)
    verilog_basename = split_filename(verilog_filename)[0]

    verilog_line_map_filename = get_output_filename_value(
        ctx,
        "verilog_line_map_file",
        verilog_basename + _VERILOG_LINE_MAP_TEXTPROTO_FILE_EXTENSION,
    )
    verilog_line_map_file = ctx.actions.declare_file(verilog_line_map_filename)
    my_generated_files.append(verilog_line_map_file)
    final_args += " --output_verilog_line_map_path={}".format(verilog_line_map_file.path)

    schedule_file = None
    if not uses_combinational_generator:
        # Pipeline generator produces a schedule artifact.
        schedule_filename = get_output_filename_value(
            ctx,
            "schedule_file",
            verilog_basename + _SCHEDULE_TEXTPROTO_FILE_EXTENSION,
        )

        schedule_file = ctx.actions.declare_file(schedule_filename)
        my_generated_files.append(schedule_file)
        final_args += " --output_schedule_path={}".format(schedule_file.path)

    verilog_file = ctx.actions.declare_file(verilog_filename)
    module_sig_filename = get_output_filename_value(
        ctx,
        "module_sig_file",
        verilog_basename + _SIGNATURE_TEXTPROTO_FILE_EXTENSION,
    )
    module_sig_file = ctx.actions.declare_file(module_sig_filename)
    my_generated_files += [verilog_file, module_sig_file]
    final_args += " --output_verilog_path={}".format(verilog_file.path)
    final_args += " --output_signature_path={}".format(module_sig_file.path)
    schedule_ir_filename = get_output_filename_value(
        ctx,
        "schedule_ir_file",
        verilog_basename + _SCHEDULE_IR_FILE_EXTENSION,
    )
    schedule_ir_file = ctx.actions.declare_file(schedule_ir_filename)
    final_args += " --output_schedule_ir_path={}".format(schedule_ir_file.path)
    my_generated_files.append(schedule_ir_file)
    block_ir_filename = get_output_filename_value(
        ctx,
        "block_ir_file",
        verilog_basename + _BLOCK_IR_FILE_EXTENSION,
    )
    block_ir_file = ctx.actions.declare_file(block_ir_filename)
    final_args += " --output_block_ir_path={}".format(block_ir_file.path)
    my_generated_files.append(block_ir_file)

    # FDO required options setup
    if uses_fdo:
        yosys_tool = ctx.executable.yosys_tool
        sta_tool = ctx.executable.sta_tool
        synth_lib = ctx.attr.standard_cells[StandardCellInfo].default_corner.liberty
        final_args += " --fdo_synthesizer_name={}".format(codegen_args.get("fdo_synthesizer_name", "yosys"))
        final_args += " --fdo_yosys_path={}".format(yosys_tool.path)
        final_args += " --fdo_sta_path={}".format(sta_tool.path)
        final_args += " --fdo_synthesis_libraries={}".format(synth_lib.path)
    else:
        yosys_tool = None
        sta_tool = None
        synth_lib = None

    # Get runfiles
    codegen_tool_runfiles = ctx.attr._xls_codegen_tool[DefaultInfo].default_runfiles

    runfiles_list = [src] + original_input_files
    if ctx.file.codegen_options_proto:
        final_args += " --codegen_options_proto={}".format(ctx.file.codegen_options_proto.path)
        runfiles_list.append(ctx.file.codegen_options_proto)

    if ctx.file.scheduling_options_proto:
        final_args += " --scheduling_options_proto={}".format(ctx.file.scheduling_options_proto.path)
        runfiles_list.append(ctx.file.scheduling_options_proto)

    if uses_fdo:
        dont_use_args = ""
        or_config = ctx.attr.standard_cells[StandardCellInfo].open_road_configuration
        if or_config:
            for dont_use_pattern in or_config.do_not_use_cell_list:
                dont_use_args += " -dont_use {} ".format(dont_use_pattern)

        runfiles_list.append(synth_lib)
        yosys_runfiles_dir = ctx.executable.yosys_tool.path + ".runfiles"
        opensta_runfiles_dir = ctx.executable.sta_tool.path + ".runfiles"
        env = {
            "ABC": yosys_runfiles_dir + "/edu_berkeley_abc/abc",
            "DONT_USE_ARGS": dont_use_args,
            "YOSYS_DATDIR": yosys_runfiles_dir + "/" + "at_clifford_yosys/techlibs/",
            "TCL_LIBRARY": opensta_runfiles_dir + "/tk_tcl/library",
        }
    else:
        env = {}

    runfiles = get_runfiles_for_xls(ctx, [codegen_tool_runfiles], runfiles_list)

    if uses_fdo:
        tools = [codegen_tool, yosys_tool, sta_tool]
    else:
        tools = [codegen_tool]

    ctx.actions.run_shell(
        outputs = my_generated_files,
        tools = tools,
        inputs = runfiles.files,
        command = "{} {} {}".format(
            codegen_tool.path,
            src.path,
            final_args,
        ),
        env = env,
        mnemonic = "Codegen",
        progress_message = "Building Verilog file: %s" % (verilog_file.path),
        toolchain = None,
    )
    return [
        CodegenInfo(
            verilog_file = verilog_file,
            module_sig_file = module_sig_file,
            verilog_line_map_file = verilog_line_map_file,
            schedule_file = schedule_file,
            schedule_ir_file = schedule_ir_file,
            block_ir_file = block_ir_file,
            delay_model = codegen_args.get("delay_model"),
            top = codegen_args.get("module_name", codegen_args.get("top")),
            pipeline_stages = codegen_args.get("pipeline_stages"),
            clock_period_ps = codegen_args.get("clock_period_ps"),
            use_fdo = codegen_args.get("use_fdo"),
            fdo_iteration_number = codegen_args.get("fdo_iteration_number"),
            fdo_delay_driven_path_number = codegen_args.get("fdo_delay_driven_path_number"),
            fdo_fanout_driven_path_number = codegen_args.get("fdo_fanout_driven_path_number"),
            fdo_refinement_stochastic_ratio = codegen_args.get("fdo_refinement_stochastic_ratio"),
            fdo_path_evaluate_strategy = codegen_args.get("fdo_path_evaluate_strategy"),
        ),
        my_generated_files,
        runfiles,
    ]

def _xls_ir_verilog_fdo_impl_wrapper(ctx):
    """The implementation of the 'xls_ir_verilog_fdo' rule.

    Wrapper for xls_ir_verilog_fdo_impl. See: xls_ir_verilog_fdo_impl.

    Args:
      ctx: The current rule's context object.
    Returns:
      CodegenInfo provider
      DefaultInfo provider
    """
    codegen_info, built_files_list, runfiles = xls_ir_verilog_fdo_impl(
        ctx,
        get_src_ir_for_xls(ctx),
        get_original_input_files_for_xls(ctx),
    )

    return [
        codegen_info,
        DefaultInfo(
            files = depset(
                direct = built_files_list,
                transitive = get_transitive_built_files_for_xls(
                    ctx,
                    [ctx.attr.src],
                ),
            ),
            runfiles = runfiles,
        ),
    ]

xls_ir_verilog_fdo = rule(
    doc = """A build rule that generates a Verilog file from an IR file using
FDO (feedback-directed optimization).  Codegen args to activate FDO
and provide required dependencies are automatically provided.  Default
values for FDO parameters are provided but can be overridden in 
"codegen_args {...}".

In FDO mode, the codegen_arg "clock_period_ps" MUST be provided.

Example:

    ```
    xls_ir_verilog_fdo(
        name = "a_verilog",
        src = "a.ir",
        codegen_args = {
            "clock_period_ps": "750",
            "fdo_iteration_number": "5",
            ...
        },
    )
    ```
    """,
    implementation = _xls_ir_verilog_fdo_impl_wrapper,
    attrs = dicts.add(
        xls_ir_common_attrs,
        xls_ir_verilog_attrs,
        CONFIG["xls_outs_attrs"],
        xls_toolchain_attrs,
    ),
)
