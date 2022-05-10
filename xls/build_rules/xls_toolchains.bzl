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

_DEFAULT_AOT_COMPILER_TARGET = "//xls/jit:aot_compiler"

_DEFAULT_INTERPRETER_TARGET = "//xls/dslx:interpreter_main"

_DEFAULT_IR_CONVERTER_TARGET = "//xls/dslx:ir_converter_main"

_DEFAULT_OPT_IR_TARGET = "//xls/tools:opt_main"

_DEFAULT_IR_EQUIVALENCE_TARGET = "//xls/tools:check_ir_equivalence_main"

_DEFAULT_EVAL_IR_TARGET = "//xls/tools:eval_ir_main"

_DEFAULT_BENCHMARK_TARGET = "//xls/tools:benchmark_main"

_DEFAULT_BENCHMARK_CODEGEN_TARGET = "//xls/tools:benchmark_codegen_main"

_DEFAULT_CODEGEN_TARGET = "//xls/tools:codegen_main"

_DEFAULT_JIT_WRAPPER_TARGET = "//xls/jit:jit_wrapper_generator_main"

_DEFAULT_XLS_TOOLCHAIN_TARGET = "//xls/build_rules:default_xls_toolchain"

def get_xls_toolchain_info(ctx):
    """Returns the XlsToolchainInfo provider of the context.

    Args:
      ctx: The current rule's context object.

    Returns:
      Returns the XlsToolchainInfo provider of the context. If the context does
      not contain the xls_toolchain attribute, returns None.
    """
    if not hasattr(ctx.attr, "_xls_toolchain"):
        return None
    return ctx.attr._xls_toolchain[XlsToolchainInfo]

def get_executable_from(target):
    """Returns the executable file of the target.

    Args:
      target: The target to extract the executable file.

    Returns:
      Returns the executable file of the target. If an executable file does not
      exist, an error is thrown.
    """
    if type(target) != "Target":
        fail("Argument 'target' from macro 'get_executable_from' must be " +
             "of 'Target' type.")
    files_list = target.files.to_list()
    if len(files_list) != 1:
        fail("Target does not contain a single file.")
    return files_list[0]

def get_runfiles_from(target):
    """Returns the runfiles of the target.

    Args:
      target: The target to extract the runfiles.

    Returns:
      Returns the runfiles of the target.
    """
    if type(target) != "Target":
        fail("Argument 'target' from macro 'get_runfiles_from' must be " +
             "of 'Target' type.")
    return target[DefaultInfo].default_runfiles

XlsToolchainInfo = provider(
    doc = "A provider containing toolchain information.",
    fields = {
        "aot_compiler_tool": "Target: the ahead-of-time compiler executable.",
        "benchmark_ir_tool": "Target: The target of the benchmark IR " +
                             "executable.",
        "benchmark_codegen_tool": "Target: The target of the benchmark " +
                                  "Verilog executable.",
        "codegen_tool": "Target: The target of the codegen executable.",
        "dslx_interpreter_tool": "Target: The target of the DSLX interpreter " +
                                 "executable.",
        "jit_wrapper_tool": "Target: The target of the JIT wrapper executable.",
        "ir_converter_tool": "Target: The target of the IR converter " +
                             "executable.",
        "ir_equivalence_tool": "Target: The target of the IR equivalence " +
                               "executable.",
        "ir_eval_tool": "Target: The target of the IR interpreter executable.",
        "opt_ir_tool": "Target: The target of the IR optimizer executable.",
    },
)

def _xls_toolchain_impl(ctx):
    xls_toolchain_info = XlsToolchainInfo(
        aot_compiler_tool = ctx.attr.aot_compiler_tool,
        benchmark_ir_tool = ctx.attr.benchmark_ir_tool,
        benchmark_codegen_tool = ctx.attr.benchmark_codegen_tool,
        codegen_tool = ctx.attr.codegen_tool,
        dslx_interpreter_tool = ctx.attr.dslx_interpreter_tool,
        jit_wrapper_tool = ctx.attr.jit_wrapper_tool,
        ir_converter_tool = ctx.attr.ir_converter_tool,
        ir_equivalence_tool = ctx.attr.ir_equivalence_tool,
        ir_eval_tool = ctx.attr.ir_eval_tool,
        opt_ir_tool = ctx.attr.opt_ir_tool,
    )
    return [xls_toolchain_info]

xls_toolchain = rule(
    doc = """A rule that returns an XlsToolchainInfo containing toolchain information.

Examples:

1. User-defined toolchain with a modified DSLX interpreter tool.

    ```
    cc_binary(
        name = "custom_dslx_interpreter_tool",
        srcs = [...],
        deps = [...],
    )

    xls_toolchain(
        name = "user_defined_xls_toolchain",
        dslx_interpreter_tool = ":custom_dslx_interpreter_tool",
    )

    xls_dslx_library(
        name = "a_user_defined_xls_toolchain_dslx",
        srcs = [
            "a.x",
        ],
        xls_toolchain = ":user_defined_xls_toolchain",
    )
    ```
    """,
    implementation = _xls_toolchain_impl,
    provides = [XlsToolchainInfo],
    attrs = {
        "aot_compiler_tool": attr.label(
            doc = "The target of the AOT IR compiler executable.",
            default = Label(_DEFAULT_AOT_COMPILER_TARGET),
            allow_single_file = True,
            executable = True,
            cfg = "exec",
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
        "benchmark_codegen_tool": attr.label(
            doc = "The target of the benchmark codegen executable.",
            default = Label(_DEFAULT_BENCHMARK_CODEGEN_TARGET),
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
    "_xls_toolchain": attr.label(
        doc = "The XLS toolchain target.",
        providers = [XlsToolchainInfo],
        default = Label(_DEFAULT_XLS_TOOLCHAIN_TARGET),
    ),
}
