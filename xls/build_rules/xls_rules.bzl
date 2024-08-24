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

"""
This module contains build rules for XLS.
"""

load("@rules_hdl//pdk:build_defs.bzl", "StandardCellInfo")
load("@bazel_skylib//lib:dicts.bzl", "dicts")
load(
    "//xls/build_rules:xls_codegen_rules.bzl",
    "xls_ir_verilog_attrs",
    "xls_ir_verilog_impl",
)
load(
    "//xls/build_rules:xls_common_rules.bzl",
    "get_transitive_built_files_for_xls",
)
load("//xls/build_rules:xls_config_rules.bzl", "CONFIG")
load(
    "//xls/build_rules:xls_dslx_rules.bzl",
    "get_dslx_test_cmd",
    "xls_dslx_test_common_attrs",
)
load(
    "//xls/build_rules:xls_ir_rules.bzl",
    "get_benchmark_ir_cmd",
    "get_eval_ir_test_cmd",
    "get_ir_equivalence_test_cmd",
    "xls_benchmark_ir_attrs",
    "xls_dslx_ir_attrs",
    "xls_dslx_ir_impl",
    "xls_eval_ir_test_attrs",
    "xls_ir_equivalence_test_attrs",
    "xls_ir_opt_ir_attrs",
    "xls_ir_opt_ir_impl",
    "xls_ir_top_attrs",
)
load(
    "//xls/build_rules:xls_providers.bzl",
    "CodegenInfo",
    "ConvIrInfo",
    "DslxInfo",
    "IrFileInfo",
    "OptIrArgInfo",
)
load("//xls/build_rules:xls_toolchains.bzl", "xls_toolchain_attrs")

_xls_dslx_opt_ir_attrs = dicts.add(
    xls_dslx_ir_attrs,
    xls_ir_opt_ir_attrs,
    CONFIG["xls_outs_attrs"],
    xls_toolchain_attrs,
)

def _xls_dslx_opt_ir_impl(ctx):
    """The implementation of the 'xls_dslx_opt_ir' rule.

    Converts a DSLX file to an IR and optimizes the IR.

    Args:
      ctx: The current rule's context object.
    Returns:
      A tuple with the following elements in the order presented:
        1. The IrFileInfo of the resulting opt ir.
        1. The DslxInfo provider
        1. The ConvIrInfo provider
        1. The OptIrArgInfo provider
        1. The list of built files.
        1. The runfiles.
    """
    dslx_conv_result_ir, dslx_info, ir_conv_info, ir_built_files, ir_runfiles = (
        xls_dslx_ir_impl(ctx)
    )
    opt_result_ir, ir_opt_info, opt_ir_built_files, opt_ir_runfiles = xls_ir_opt_ir_impl(
        ctx,
        dslx_conv_result_ir,
        ir_conv_info.original_input_files,
    )
    return [
        opt_result_ir,
        dslx_info,
        ir_conv_info,
        ir_opt_info,
        ir_built_files + opt_ir_built_files,
        ir_runfiles.merge(opt_ir_runfiles),
    ]

def _xls_dslx_opt_ir_impl_wrapper(ctx):
    """The implementation of the 'xls_dslx_opt_ir' rule.

    Converts a DSLX file to an IR and optimizes the IR.

    Args:
      ctx: The current rule's context object.
    Returns:
      IrFileInfo provider.
      DslxInfo provider.
      ConvIrInfo provider.
      OptIrArgInfo provider.
      DefaultInfo provider.
    """
    result_ir, dslx_info, ir_conv_info, ir_opt_info, built_files, runfiles = (
        _xls_dslx_opt_ir_impl(ctx)
    )
    return [
        result_ir,
        dslx_info,
        ir_conv_info,
        ir_opt_info,
        DefaultInfo(
            files = depset(
                direct = built_files,
                transitive = get_transitive_built_files_for_xls(ctx),
            ),
            runfiles = runfiles,
        ),
    ]

xls_dslx_opt_ir = rule(
    doc = """A build rule that generates an optimized IR file from a DSLX source file.

Examples:

1. A simple example.

    ```
    # Assume a xls_dslx_library target bc_dslx is present.
    xls_dslx_opt_ir(
        name = "d_opt_ir",
        srcs = ["d.x"],
        deps = [":bc_dslx"],
        dslx_top = "d",
    )
    ```
    """,
    implementation = _xls_dslx_opt_ir_impl_wrapper,
    attrs = _xls_dslx_opt_ir_attrs,
)

def _xls_dslx_opt_ir_test_impl(ctx):
    """The implementation of the 'xls_dslx_opt_ir_test' rule.

    Executes the test commands for the following rules in the order presented:
      1) xls_dslx_test
      2) xls_ir_equivalence_test
      3) xls_eval_ir_test
      4) xls_benchmark_ir

    Args:
      ctx: The current rule's context object.
    Returns:
      DefaultInfo provider
    """
    dslx_test_file = ctx.attr.dep[DslxInfo].target_dslx_source_files
    conv_ir_file = ctx.attr.dep[OptIrArgInfo].unopt_ir
    opt_ir_file = ctx.attr.dep[CodegenInfo].input_ir if CodegenInfo in ctx.attr.dep else ctx.attr.dep[IrFileInfo]
    runfiles = ctx.runfiles(files = ctx.attr.dep[DslxInfo].dslx_source_files
        .to_list())

    # xls_dslx_test
    my_runfiles, dslx_test_cmd = get_dslx_test_cmd(ctx, dslx_test_file)
    dslx_test_cmd = "\n".join(dslx_test_cmd)
    runfiles = runfiles.merge(my_runfiles)

    # xls_ir_equivalence_test
    my_runfiles, ir_equivalence_test_cmd = get_ir_equivalence_test_cmd(
        ctx,
        conv_ir_file,
        opt_ir_file,
        True,
    )
    runfiles = runfiles.merge(my_runfiles)

    # xls_eval_ir_test
    my_runfiles, eval_ir_test_cmd = get_eval_ir_test_cmd(
        ctx,
        conv_ir_file,
    )
    runfiles = runfiles.merge(my_runfiles)

    # xls_benchmark_ir
    my_runfiles, benchmark_ir_cmd = get_benchmark_ir_cmd(
        ctx,
        conv_ir_file,
    )
    runfiles = runfiles.merge(my_runfiles)

    executable_file = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(
        output = executable_file,
        content = "\n".join([
            "#!/usr/bin/env bash",
            "set -e",
            dslx_test_cmd,
            ir_equivalence_test_cmd,
            eval_ir_test_cmd,
            benchmark_ir_cmd,
            "exit 0",
        ]),
        is_executable = True,
    )
    return [
        DefaultInfo(
            runfiles = runfiles,
            files = depset(
                direct = [executable_file],
                transitive = get_transitive_built_files_for_xls(
                    ctx,
                    [ctx.attr.dep],
                ),
            ),
            executable = executable_file,
        ),
    ]

_xls_dslx_opt_ir_test_impl_attrs = {
    "dep": attr.label(
        doc = "The xls_dslx_opt_ir target to test.",
        providers = [
            DslxInfo,
            ConvIrInfo,
            OptIrArgInfo,
        ],
    ),
}

xls_dslx_opt_ir_test = rule(
    doc = """A build rule that tests a xls_dslx_opt_ir target.

Executes the test commands for the following rules in the order presented:

1. xls_dslx_test
1. xls_ir_equivalence_test
1. xls_eval_ir_test
1. xls_benchmark_ir

Examples:

1. A simple example.

    ```
    xls_dslx_opt_ir(
        name = "a_opt_ir",
        srcs = ["a.x"],
        dslx_top = "a",
    )

    xls_dslx_opt_ir_test(
        name = "a_opt_ir_test",
        dep = ":a_opt_ir",
    )
    ```
    """,
    implementation = _xls_dslx_opt_ir_test_impl,
    attrs = dicts.add(
        _xls_dslx_opt_ir_test_impl_attrs,
        xls_dslx_test_common_attrs,
        xls_ir_equivalence_test_attrs,
        xls_eval_ir_test_attrs,
        xls_benchmark_ir_attrs,
        xls_ir_top_attrs,
        xls_toolchain_attrs,
    ),
    test = True,
)

def _xls_dslx_verilog_impl(ctx):
    """The implementation of the 'xls_dslx_verilog' rule.

    Converts a DSLX file to an IR, optimizes the IR, and generates a verilog
    file from the optimized IR.

    Args:
      ctx: The current rule's context object.
    Returns:
      DslxInfo provider.
      ConvIRInfo provider.
      OptIRInfo provider.
      CodegenInfo provider.
      DefaultInfo provider.
    """
    (
        ir_result,
        dslx_info,
        ir_conv_info,
        ir_opt_info,
        built_files,
        runfiles,
    ) = _xls_dslx_opt_ir_impl(ctx)
    codegen_info, verilog_built_files, verilog_runfiles = xls_ir_verilog_impl(
        ctx,
        ir_result,
        ir_conv_info,
    )
    return [
        dslx_info,
        ir_conv_info,
        ir_opt_info,
        codegen_info,
        DefaultInfo(
            files = depset(
                direct = built_files + verilog_built_files,
                transitive = get_transitive_built_files_for_xls(ctx),
            ),
            runfiles = runfiles.merge(verilog_runfiles),
        ),
    ]

_dslx_verilog_attrs = dicts.add(
    xls_dslx_ir_attrs,
    xls_ir_opt_ir_attrs,
    xls_ir_verilog_attrs,
    CONFIG["xls_outs_attrs"],
    xls_toolchain_attrs,
)

xls_dslx_verilog = rule(
    doc = """A build rule that generates a Verilog file from a DSLX source file.

Examples:

1. A simple example.

    ```
    # Assume a xls_dslx_library target bc_dslx is present.
    xls_dslx_verilog(
        name = "d_verilog",
        srcs = ["d.x"],
        deps = [":bc_dslx"],
        codegen_args = {
            "pipeline_stages": "1",
        },
        dslx_top = "d",
    )
    ```
    """,
    implementation = _xls_dslx_verilog_impl,
    attrs = _dslx_verilog_attrs,
)

def _xls_delay_model_generation_impl(ctx):
    script_contents = [""]

    standard_cell = ctx.attr.standard_cells[StandardCellInfo]

    if not standard_cell.default_corner:
        fail("No default corner found on " + str(ctx.attr.standard_cell))
    lib = standard_cell.default_corner.liberty

    yosys = ctx.executable._yosys
    yosys_runfiles_dir = yosys.path + ".runfiles"

    sta = ctx.executable._opensta
    sta_runfiles_dir = sta.path + ".runfiles"

    run_timing_characterization = ctx.executable._run_timing_characterization
    timing_characterization_client_main = ctx.executable._timing_characterization_client_main
    yosys_server_main = ctx.executable._yosys_server_main

    samples_file = ctx.file.samples_file

    default_driver_cell = getattr(standard_cell, "default_input_driver_cell", "")
    default_load = getattr(standard_cell, "default_output_load", "")

    script_contents.append("export YOSYS_DATDIR={}/at_clifford_yosys/techlibs/".format(yosys_runfiles_dir))
    script_contents.append("export ABC={}/edu_berkeley_abc/abc".format(yosys_runfiles_dir))
    script_contents.append("export TCL_LIBRARY={}/tk_tcl/library".format(sta_runfiles_dir))
    script_contents.append("export DONT_USE_ARGS=")
    script_contents.append("set -e")

    cmd = [run_timing_characterization.short_path]
    cmd.append("--yosys_path \"{}\"".format(yosys.short_path))
    cmd.append("--sta_path \"{}\"".format(sta.short_path))
    cmd.append("--synth_libs \"{}\"".format(lib.short_path))
    cmd.append("--default_driver_cell \"{}\"".format(default_driver_cell))
    cmd.append("--default_load \"{}\"".format(default_load))
    cmd.append("--client \"{}\"".format(timing_characterization_client_main.short_path))
    cmd.append("--server \"{}\"".format(yosys_server_main.short_path))
    cmd.append("--samples_path \"{}\"".format(samples_file.short_path))
    cmd.append("--out_path \"${{BUILD_WORKING_DIRECTORY}}/{}\"".format(ctx.attr.name + ".textproto"))
    cmd.append("\"$@\"")

    script_str = "\n".join(script_contents) + "\n" + " ".join(cmd) + "\n"

    script = ctx.actions.declare_file(ctx.attr.name + ".sh")
    ctx.actions.write(
        output = script,
        content = script_str,
        is_executable = True,
    )

    transitive_runfiles = [
        ctx.attr.standard_cells[DefaultInfo].default_runfiles,
        ctx.attr._opensta[DefaultInfo].default_runfiles,
        ctx.attr._run_timing_characterization[DefaultInfo].default_runfiles,
        ctx.attr._timing_characterization_client_main[DefaultInfo].default_runfiles,
        ctx.attr._yosys[DefaultInfo].default_runfiles,
        ctx.attr._yosys_server_main[DefaultInfo].default_runfiles,
    ]

    return [
        DefaultInfo(
            files = depset(direct = [script]),
            runfiles = ctx.runfiles(files = [script, samples_file]).merge_all(transitive_runfiles),
            executable = script,
        ),
    ]

xls_delay_model_generation = rule(
    implementation = _xls_delay_model_generation_impl,
    doc = """Builds a script to generate an XLS delay model for one PDK corner.

This rule gathers the locations of the required dependencies
(Yosys, OpenSTA, helper scripts, and cell libraries) and
generates a wrapper script that invokes "run_timing_characterization"
with the dependency locations provided as args.

Any extra runtime args will get passed in to the
"run_timing_characterization" script (e.g. "--debug" or "--quick_run").

The script must be "run" from the root of the workspace
to perform the timing characterization.  The output textproto
will be produced in the current directory (which, as just
stated, must be the root of the workspace).

Currently, only a subset of XLS operators are characterized,
including most arithmetic, logical, and shift operators.
However, many common operators such as "concat", "bit_slice",
and "encode" are missing, and so the delay model that is
currently produced should be considered INCOMPLETE.""",
    executable = True,
    attrs = {
        "standard_cells": attr.label(
            doc = "Target for the PDK; will use the target's default corner.",
            providers = [StandardCellInfo],
        ),
        "samples_file": attr.label(
            doc = "Proto providing sample points.",
            allow_single_file = True,
        ),
        "_opensta": attr.label(
            default = Label("@org_theopenroadproject//:opensta"),
            executable = True,
            cfg = "target",
        ),
        "_run_timing_characterization": attr.label(
            default = Label("//xls/tools:run_timing_characterization"),
            executable = True,
            cfg = "target",
        ),
        "_timing_characterization_client_main": attr.label(
            default = Label("//xls/synthesis:timing_characterization_client_main"),
            executable = True,
            cfg = "target",
        ),
        "_yosys": attr.label(
            default = Label("@at_clifford_yosys//:yosys"),
            executable = True,
            cfg = "target",
        ),
        "_yosys_server_main": attr.label(
            default = Label("//xls/synthesis/yosys:yosys_server_main"),
            executable = True,
            cfg = "target",
        ),
    },
)
