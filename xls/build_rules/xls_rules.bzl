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

load(
    "//xls/build_rules:xls_dslx_rules.bzl",
    "DslxFilesInfo",
    "dslx_exec_attrs",
    "dslx_test_common_attrs",
    "get_dslx_test_cmd",
    "get_transitive_dslx_dummy_files_depset",
    "get_transitive_dslx_srcs_files_depset",
)
load(
    "//xls/build_rules:xls_ir_rules.bzl",
    "IRConvInfo",
    "IROptInfo",
    "dslx_to_ir_attrs",
    "dslx_to_ir_impl",
    "get_ir_benchmark_cmd",
    "get_ir_equivalence_test_cmd",
    "get_ir_eval_test_cmd",
    "ir_benchmark_attrs",
    "ir_equivalence_test_attrs",
    "ir_eval_test_attrs",
    "ir_opt_attrs",
    "ir_opt_impl",
)
load(
    "//xls/build_rules:xls_codegen_rules.bzl",
    "ir_to_codegen_attrs",
    "ir_to_codegen_impl",
)
load("@bazel_skylib//lib:dicts.bzl", "dicts")

def _dslx_to_ir_opt_impl(ctx, src):
    """The implementation of the 'dslx_to_ir_opt' rule.

    Converts a DSLX file to an IR and optimizes the IR.

    Args:
      ctx: The current rule's context object.
      src: The source file.
    Returns:
      DslxFilesInfo provider.
      IRConvInfo provider.
      IROptInfo provider.
      DefaultInfo provider.
    """
    ir_conv_info, ir_conv_default_info = dslx_to_ir_impl(ctx, src)
    ir_opt_info, ir_opt_default_info = ir_opt_impl(
        ctx,
        ir_conv_info.ir_conv_file,
    )
    return [
        DslxFilesInfo(
            dslx_sources = get_transitive_dslx_srcs_files_depset(
                [src],
                ctx.attr.deps,
            ),
            dslx_dummy_files = get_transitive_dslx_dummy_files_depset(
                None,
                ctx.attr.deps,
            ),
        ),
        ir_conv_info,
        ir_opt_info,
        DefaultInfo(
            files = depset(
                ir_conv_default_info.files.to_list() +
                ir_opt_default_info.files.to_list(),
            ),
        ),
    ]

_dslx_to_ir_opt_attrs = dicts.add(
    dslx_to_ir_attrs,
    ir_opt_attrs,
)

def _dslx_to_ir_opt_impl_wrapper(ctx):
    """The implementation of the 'dslx_to_ir_opt' rule.

    Wrapper for _dslx_to_ir_opt_impl. See: _dslx_to_ir_opt_impl.

    Args:
      ctx: The current rule's context object.
    Returns:
      See: _dslx_to_ir_opt_impl.
    """
    return _dslx_to_ir_opt_impl(ctx, ctx.file.src)

dslx_to_ir_opt = rule(
    doc = """A build rule that generates an optimized IR file from a DSLX source file.

        Examples:

        1) Generate optimized IR from a DSLX source.

        ```
            dslx_to_ir_opt(
                name = "a_ir_opt",
                src = "a.x",
            )
        ```

        2) Generate optimized IR with dependency on dslx_library targets.

        ```
            dslx_library(
                name = "files_ab",
                srcs = [
                    "a.x",
                    "b.x",
                ],
            )

            dslx_library(
                name = "c",
                srcs = [
                    "c.x",
                ],
            )

            dslx_to_ir_opt(
                name = "d_ir_opt",
                src = "d.x",
                deps = [
                    ":files_ab",
                    ":c",
                ],
            )
        ```
    """,
    implementation = _dslx_to_ir_opt_impl_wrapper,
    attrs = _dslx_to_ir_opt_attrs,
)

def _dslx_to_ir_opt_test_impl(ctx):
    """The implementation of the 'dslx_to_ir_opt_test' rule.

    Executes the commands in the order presented in the list for the following
    rules:
      1) dslx_test
      2) ir_equivalence_test (if attribute prove_unopt_eq_opt is enabled)
      3) ir_eval_test (if attribute generate_benchmark is enabled)
      4) ir_benchmark (if attribute generate_benchmark is enabled)

    Args:
      ctx: The current rule's context object.
    Returns:
      DefaultInfo provider
    """
    dslx_source_files = ctx.attr.dep[DslxFilesInfo].dslx_sources.to_list()
    dslx_test_file = ctx.attr.dep[IRConvInfo].dslx_source_file
    ir_conv_file = ctx.attr.dep[IRConvInfo].ir_conv_file
    ir_opt_file = ctx.attr.dep[IROptInfo].ir_opt_file
    runfiles = dslx_source_files

    # dslx_test
    my_runfiles, dslx_test_cmd = get_dslx_test_cmd(ctx, dslx_test_file)
    runfiles += my_runfiles

    # ir_equivalence_test
    my_runfiles, ir_equivalence_test_cmd = get_ir_equivalence_test_cmd(
        ctx,
        ir_conv_file,
        ir_opt_file,
    )
    runfiles += my_runfiles

    # ir_eval_test
    my_runfiles, ir_eval_test_cmd = get_ir_eval_test_cmd(
        ctx,
        ir_conv_file,
    )
    runfiles += my_runfiles

    # ir_benchmark
    my_runfiles, ir_benchmark_cmd = get_ir_benchmark_cmd(
        ctx,
        ir_conv_file,
    )
    runfiles += my_runfiles

    executable_file = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(
        output = executable_file,
        content = "\n".join([
            "#!/bin/bash",
            "set -e",
            dslx_test_cmd,
            ir_equivalence_test_cmd if ctx.attr.prove_unopt_eq_opt else "",
            ir_eval_test_cmd if ctx.attr.generate_benchmark else "",
            ir_benchmark_cmd if ctx.attr.generate_benchmark else "",
            "exit 0",
        ]),
        is_executable = True,
    )
    return [
        DefaultInfo(
            runfiles = ctx.runfiles(files = runfiles),
            files = depset([executable_file]),
            executable = executable_file,
        ),
    ]

_dslx_to_ir_opt_test_impl_attrs = {
    "dep": attr.label(
        doc = "The dslx_to_ir_opt target to test.",
        providers = [
            DslxFilesInfo,
            IRConvInfo,
            IROptInfo,
        ],
    ),
    "prove_unopt_eq_opt": attr.bool(
        doc = "Whether or not to generate a test to compare semantics of opt" +
              "vs. non-opt IR.",
        default = True,
    ),
    "generate_benchmark": attr.bool(
        doc = "Whether or not to create a benchmark target (that analyses " +
              "XLS scheduled critical path).",
        default = True,
    ),
}

dslx_to_ir_opt_test = rule(
    doc = """A build rule that tests a dslx_to_ir_opt target.

        Example:
        ```
            dslx_to_ir_opt(
                name = "a_ir_opt",
                src = "a.x",
            )

            dslx_to_ir_opt_test(
                name = "a_ir_opt_test",
                dep = ":a_ir_opt",
            )
        ```
    """,
    implementation = _dslx_to_ir_opt_test_impl,
    attrs = dicts.add(
        _dslx_to_ir_opt_test_impl_attrs,
        dslx_exec_attrs,
        dslx_test_common_attrs,
        ir_equivalence_test_attrs,
        ir_eval_test_attrs,
        ir_benchmark_attrs,
    ),
    test = True,
)

def _dslx_to_codegen_impl(ctx):
    """The implementation of the 'dslx_to_codegen' rule.

    Converts a DSLX file to an IR, optimizes the IR, and generates a verilog
    file from the optimized IR.

    Args:
      ctx: The current rule's context object.
    Returns:
      DslxFilesInfo provider.
      IRConvInfo provider.
      IROptInfo provider.
      CodegenInfo provider.
      DefaultInfo provider.
    """
    dslx_test_file = ctx.file.src
    dslx_files_info, ir_conv_info, ir_opt_info, dslx_to_ir_default_info = _dslx_to_ir_opt_impl(
        ctx,
        dslx_test_file,
    )
    codegen_info, codegen_default_info = ir_to_codegen_impl(
        ctx,
        ir_opt_info.ir_opt_file,
    )
    return [
        dslx_files_info,
        ir_conv_info,
        ir_opt_info,
        codegen_info,
        DefaultInfo(
            files = depset(
                dslx_to_ir_default_info.files.to_list() +
                codegen_default_info.files.to_list(),
            ),
        ),
    ]

_dslx_to_codegen_attrs = dicts.add(
    dslx_to_ir_attrs,
    ir_opt_attrs,
    ir_to_codegen_attrs,
)

dslx_to_codegen = rule(
    doc = """A build rule that generates a Verilog file from a DSLX source file.

        Examples:

        1) Generate Verilog from a DSLX source.

        ```
            dslx_to_codegen(
                name = "a_verilog",
                src = "a.x",
                codegen_args = {
                    "pipeline_stages": "1",
                },
            )
        ```

        2) Generate Verilog with dependency on dslx_library targets.

        ```
            dslx_library(
                name = "files_ab",
                srcs = [
                    "a.x",
                    "b.x",
                ],
            )

            dslx_library(
                name = "c",
                srcs = [
                    "c.x",
                ],
            )

            dslx_to_codegen(
                name = "d_verilog",
                src = "d.x",
                deps = [
                    ":files_ab",
                    ":c",
                ],
                codegen_args = {
                    "pipeline_stages": "1",
                },
            )
        ```
    """,
    implementation = _dslx_to_codegen_impl,
    attrs = _dslx_to_codegen_attrs,
)

# TODO(vmirian) 2021-05-20 When https://github.com/google/xls/issues/418 and
# https://github.com/google/xls/issues/419 are resolved:
# implement dslx_to_codegen_test
