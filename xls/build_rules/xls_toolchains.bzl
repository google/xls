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

_DEFAULT_AOT_BASIC_FUNCTION_TARGET = "//xls/jit:aot_basic_function_entrypoint_main"

_DEFAULT_INTERPRETER_TARGET = "//xls/dslx:interpreter_main"

# Note: exported so we can use it in our macro implementation (which does not
# get a toolchain ctx).
_DEFAULT_DSLX_FMT_TARGET = "//xls/dslx:dslx_fmt"
DEFAULT_DSLX_FMT_TARGET = _DEFAULT_DSLX_FMT_TARGET

_DEFAULT_IR_CONVERTER_TARGET = "//xls/dslx/ir_convert:ir_converter_main"

_DEFAULT_OPT_IR_TARGET = "//xls/tools:opt_main"

_DEFAULT_IR_EQUIVALENCE_TARGET = "//xls/tools:check_ir_equivalence_main"

_DEFAULT_EVAL_IR_TARGET = "//xls/tools:eval_ir_main"

_DEFAULT_BENCHMARK_TARGET = "//xls/tools:benchmark_main"

_DEFAULT_BENCHMARK_CODEGEN_TARGET = "//xls/tools:benchmark_codegen_main"

_DEFAULT_CODEGEN_TARGET = "//xls/tools:codegen_main"

_DEFAULT_JIT_WRAPPER_TARGET = "//xls/jit:jit_wrapper_generator_main"

xls_toolchain_attrs = {
    "_xls_aot_compiler_tool": attr.label(
        doc = "The target of the AOT IR compiler executable.",
        default = Label(_DEFAULT_AOT_COMPILER_TARGET),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
    "_xls_aot_basic_function_tool": attr.label(
        default = Label(_DEFAULT_AOT_BASIC_FUNCTION_TARGET),
        executable = True,
        cfg = "exec",
        allow_files = True,
    ),
    "_xls_dslx_interpreter_tool": attr.label(
        doc = "The target of the DSLX interpreter executable.",
        default = Label(_DEFAULT_INTERPRETER_TARGET),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
    "_xls_dslx_autoformat_tool": attr.label(
        doc = "The target of the DSLX auto-format executable.",
        default = Label(DEFAULT_DSLX_FMT_TARGET),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
    "_xls_ir_converter_tool": attr.label(
        doc = "The target of the IR converter executable.",
        default = Label(_DEFAULT_IR_CONVERTER_TARGET),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
    "_xls_opt_ir_tool": attr.label(
        doc = "The target of the IR optimizer executable.",
        default = Label(_DEFAULT_OPT_IR_TARGET),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
    "_xls_ir_equivalence_tool": attr.label(
        doc = "The target of the IR equivalence executable.",
        default = Label(_DEFAULT_IR_EQUIVALENCE_TARGET),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
    "_xls_ir_eval_tool": attr.label(
        doc = "The target of the IR interpreter executable.",
        default = Label(_DEFAULT_EVAL_IR_TARGET),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
    "_xls_benchmark_ir_tool": attr.label(
        doc = "The target of the benchmark IR executable.",
        default = Label(_DEFAULT_BENCHMARK_TARGET),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
    "_xls_benchmark_codegen_tool": attr.label(
        doc = "The target of the benchmark codegen executable.",
        default = Label(_DEFAULT_BENCHMARK_CODEGEN_TARGET),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
    "_xls_codegen_tool": attr.label(
        doc = "The target of the codegen executable.",
        default = Label(_DEFAULT_CODEGEN_TARGET),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
    "_xls_jit_wrapper_tool": attr.label(
        doc = "The target of the JIT wrapper executable.",
        default = Label(_DEFAULT_JIT_WRAPPER_TARGET),
        allow_files = True,
        executable = True,
        cfg = "exec",
    ),
}

def _xls_toolchain_impl(ctx):
    targets = [
        ctx.attr._xls_aot_compiler_tool,
        ctx.attr._xls_aot_basic_function_tool,
        ctx.attr._xls_benchmark_ir_tool,
        ctx.attr._xls_benchmark_codegen_tool,
        ctx.attr._xls_codegen_tool,
        ctx.attr._xls_dslx_interpreter_tool,
        ctx.attr._xls_jit_wrapper_tool,
        ctx.attr._xls_ir_converter_tool,
        ctx.attr._xls_ir_equivalence_tool,
        ctx.attr._xls_ir_eval_tool,
        ctx.attr._xls_opt_ir_tool,
    ]
    files = depset(transitive = [t.files for t in targets if t != None])
    return DefaultInfo(files = files, runfiles = ctx.runfiles(transitive_files = files))

xls_toolchain = rule(
    doc = """A rule that returns a filegroup containing all the toolchain files.

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
    provides = [DefaultInfo],
    attrs = xls_toolchain_attrs,
)
