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
    "//xls/build_rules:xls_ir_rules.bzl",
    "convert_to_ir",
    "dslx_to_ir_attrs",
    "ir_opt_attrs",
    "optimize_ir",
)
load(
    "//xls/build_rules:xls_codegen_rules.bzl",
    "ir_to_codegen_attrs",
    "ir_to_codegen_impl",
)

def _dslx_to_codegen_impl(ctx):
    """The implementation of the 'dslx_to_codegen' rule.

    Converts a DSLX file to an IR, optimizes the IR, and generates a verilog
    file from the optimized IR.

    Args:
      ctx: The current rule's context object.
    Returns:
      See codegen_impl.
    """
    file = convert_to_ir(ctx, ctx.file.src)
    file = optimize_ir(ctx, file)
    return ir_to_codegen_impl(ctx, file)

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
