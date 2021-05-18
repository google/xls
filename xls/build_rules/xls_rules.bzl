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
)
load(
    "//xls/build_rules:xls_ir_rules.bzl",
    "convert_to_ir",
    "dslx_to_ir_attrs",
    "get_ir_benchmark_cmd",
    "get_ir_equivalence_test_cmd",
    "get_ir_eval_test_cmd",
    "ir_benchmark_attrs",
    "ir_equivalence_test_attrs",
    "ir_eval_test_attrs",
    "ir_opt_attrs",
    "optimize_ir",
)
load(
    "//xls/build_rules:xls_codegen_rules.bzl",
    "ir_to_codegen_attrs",
    "ir_to_codegen_impl",
)

DSLXToCodegenInfo = provider(
    doc = "A provider containing file information for a 'dslx_to_codegen' " +
          "target. It is created and returned by the dslx_to_codegen rule.",
    fields = {
        # TODO(https://github.com/google/xls/issues/392)
        # When the issue (above) is resolved, use 'dummy' files for compilation
        # and testing.
        "dslx_source_files": "List: The DSLX source files.",
        "dslx_test_file": "File: The DSLX test file.",
        "ir_conv_file": "File: The IR file converted from a DSLX source.",
        "ir_opt_file": "File: The IR optimized file.",
        "verilog_file": "File: The Verilog file.",
    },
)

def _dslx_to_codegen_impl(ctx):
    """The implementation of the 'dslx_to_codegen' rule.

    Converts a DSLX file to an IR, optimizes the IR, and generates a verilog
    file from the optimized IR.

    Args:
      ctx: The current rule's context object.
    Returns:
      DSLXToCodegenInfo provider.
    """
    dslx_test_file = ctx.file.src
    dslx_source_files = []
    for dep in ctx.attr.deps:
        dslx_source_files += dep[DslxFilesInfo].dslx_sources.to_list()
    ir_conv_file = convert_to_ir(ctx, dslx_test_file)
    ir_opt_file = optimize_ir(ctx, ir_conv_file)
    codegen_info, _ = ir_to_codegen_impl(ctx, ir_opt_file)
    verilog_file = codegen_info.verilog_file
    return [
        DSLXToCodegenInfo(
            dslx_source_files = dslx_source_files,
            dslx_test_file = dslx_test_file,
            ir_conv_file = ir_conv_file,
            ir_opt_file = ir_opt_file,
            verilog_file = verilog_file,
        ),
        DefaultInfo(files = depset([
            dslx_test_file,
            ir_conv_file,
            ir_opt_file,
            verilog_file,
        ])),
    ]

_dslx_to_codegen_attrs = dict(dslx_to_ir_attrs.items() + ir_opt_attrs.items() +
                              ir_to_codegen_attrs.items())

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

def _dslx_to_codegen_test_impl(ctx):
    """The implementation of the 'dslx_to_codegen_test' rule.

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
    dslx_to_codegen_info = ctx.attr.dep[DSLXToCodegenInfo]
    dslx_source_files = dslx_to_codegen_info.dslx_source_files
    dslx_test_file = dslx_to_codegen_info.dslx_test_file
    ir_conv_file = dslx_to_codegen_info.ir_conv_file
    ir_opt_file = dslx_to_codegen_info.ir_opt_file
    verilog_file = dslx_to_codegen_info.verilog_file
    runfiles = list(dslx_source_files)

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

_dslx_to_codegen_test_impl_attrs = {
    "dep": attr.label(
        doc = "The dslx_to_codegen target to test.",
        providers = [DSLXToCodegenInfo],
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

dslx_to_codegen_test = rule(
    doc = """A build rule that tests a dslx_to_codegen target.

        Example:
        ```
            dslx_to_codegen(
                name = "a_verilog",
                src = "a.x",
                codegen_args = {
                    "pipeline_stages": "1",
                },
            )

            dslx_to_codegen_test(
                name = "a_verilog_test",
                dep = "a_verilog",
            )
        ```
    """,
    implementation = _dslx_to_codegen_test_impl,
    attrs = dict(
        _dslx_to_codegen_test_impl_attrs.items() +
        dslx_exec_attrs.items() +
        dslx_test_common_attrs.items() +
        ir_equivalence_test_attrs.items() +
        ir_eval_test_attrs.items() +
        ir_benchmark_attrs.items(),
    ),
    test = True,
)
