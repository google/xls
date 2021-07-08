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
This module contains toolchains for XLS.
"""

_DEFAULT_STDLIB_TARGET = "//xls/dslx/stdlib:x_files"

_DEFAULT_INTERPRETER_TARGET = "//xls/dslx:interpreter_main"

_DEFAULT_IR_CONVERTER_TARGET = "//xls/dslx:ir_converter_main"

_DEFAULT_OPT_IR_TARGET = "//xls/tools:opt_main"

_DEFAULT_IR_EQUIVALENCE_TARGET = "//xls/tools:check_ir_equivalence_main"

_DEFAULT_EVAL_IR_TARGET = "//xls/tools:eval_ir_main"

_DEFAULT_BENCHMARK_TARGET = "//xls/tools:benchmark_main"

_DEFAULT_CODEGEN_TARGET = "//xls/tools:codegen_main"

_DEFAULT_JIT_WRAPPER_TARGET = "//xls/jit:jit_wrapper_generator_main"

_DEFAULT_XLS_TOOLCHAIN_TARGET = "//xls/build_rules:default_xls_toolchain"

def get_xls_toolchain_info(ctx):
    return ctx.attr.toolchain[XlsToolchainInfo]

XlsToolchainInfo = provider(
    doc = "A provider containing toolchain information.",
    fields = {
        # Create a list and target instance for the standard library since
        # converting a depset to a list is costly. The conversion will be
        # performed once.
        "dslx_std_lib_list": "List: The list of files composing the DSLX std " +
                             "library.",
        "dslx_std_lib_target": "Target: The target of the DSLX std library.",
        "dslx_interpreter_tool": "File: The DSLX interpreter executable.",
        "ir_converter_tool": "File: The IR converter executable.",
        "opt_ir_tool": "File: The IR optimizer executable.",
        "ir_equivalence_tool": "File: The IR equivalence executable.",
        "ir_eval_tool": "File: The IR interpreter executable.",
        "benchmark_ir_tool": "File: The benchmark IR executable.",
        "codegen_tool": "File: The codegen executable.",
        "jit_wrapper_tool": "File: The JIT wrapper executable.",
    },
)

def _xls_toolchain_impl(ctx):
    xls_toolchain_info = XlsToolchainInfo(
        dslx_std_lib_list = ctx.attr.dslx_std_lib.files.to_list(),
        dslx_std_lib_target = ctx.attr.dslx_std_lib,
        dslx_interpreter_tool = (
            ctx.attr.dslx_interpreter_tool.files.to_list()[0]
        ),
        ir_converter_tool = ctx.attr.ir_converter_tool.files.to_list()[0],
        opt_ir_tool = ctx.attr.opt_ir_tool.files.to_list()[0],
        ir_equivalence_tool = ctx.attr.ir_equivalence_tool.files.to_list()[0],
        ir_eval_tool = ctx.attr.ir_eval_tool.files.to_list()[0],
        benchmark_ir_tool = ctx.attr.benchmark_ir_tool.files.to_list()[0],
        codegen_tool = ctx.attr.codegen_tool.files.to_list()[0],
        jit_wrapper_tool = ctx.attr.jit_wrapper_tool.files.to_list()[0],
    )
    return [xls_toolchain_info]

xls_toolchain = rule(
    implementation = _xls_toolchain_impl,
    provides = [XlsToolchainInfo],
    attrs = {
        "dslx_std_lib": attr.label(
            doc = "The target containing the DSLX std library.",
            default = Label(_DEFAULT_STDLIB_TARGET),
            cfg = "target",
        ),
        "dslx_interpreter_tool": attr.label(
            doc = "The target of the DSLX interpreter executable.",
            default = Label(_DEFAULT_INTERPRETER_TARGET),
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),
        "ir_converter_tool": attr.label(
            doc = "The target of the IR converter executable.",
            default = Label(_DEFAULT_IR_CONVERTER_TARGET),
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),
        "opt_ir_tool": attr.label(
            doc = "The target of the IR optimizer executable.",
            default = Label(_DEFAULT_OPT_IR_TARGET),
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),
        "ir_equivalence_tool": attr.label(
            doc = "The target of the IR equivalence executable.",
            default = Label(_DEFAULT_IR_EQUIVALENCE_TARGET),
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),
        "ir_eval_tool": attr.label(
            doc = "The target of the IR interpreter executable.",
            default = Label(_DEFAULT_EVAL_IR_TARGET),
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),
        "benchmark_ir_tool": attr.label(
            doc = "The target of the benchmark IR executable.",
            default = Label(_DEFAULT_BENCHMARK_TARGET),
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),
        "codegen_tool": attr.label(
            doc = "The target of the codegen executable.",
            default = Label(_DEFAULT_CODEGEN_TARGET),
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),
        "jit_wrapper_tool": attr.label(
            doc = "The target of the JIT wrapper executable.",
            default = Label(_DEFAULT_JIT_WRAPPER_TARGET),
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),
    },
)

xls_toolchain_attr = {
    "toolchain": attr.label(
        doc = "The toolchain target.",
        providers = [XlsToolchainInfo],
        default = Label(_DEFAULT_XLS_TOOLCHAIN_TARGET),
    ),
}
