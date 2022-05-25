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

DslxInfo = provider(
    doc = "A provider containing DSLX file information for the target. The " +
          "provider is primarily created and returned by the " +
          "xls_dslx_library rule.",
    fields = {
        "dslx_dummy_files": "Depset: A depset containing the DSLX generated " +
                            "dummy (.dummy) files of the xls_dslx_library " +
                            "target and its xls_dslx_library dependencies. " +
                            "A DSLX dummy file is generated when the DSLX " +
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

ConvIRInfo = provider(
    doc = "A provider containing IR conversion file information for the " +
          "target. It is created and returned by the xls_dslx_ir rule.",
    fields = {
        "conv_ir_file": "File: The IR file converted from a source file.",
    },
)

OptIRInfo = provider(
    doc = "A provider containing IR optimization file information for the " +
          "target. It is created and returned by the xls_ir_opt_ir rule.",
    fields = {
        "input_ir_file": "File: The IR file input file.",
        "opt_ir_args": "Dictionary: The arguments for the IR optimizer.",
        "opt_ir_file": "File: The IR optimized file.",
    },
)

CodegenInfo = provider(
    doc = "A provider containing Codegen file information for the target. It " +
          "is created and returned by the xls_ir_verilog rule.",
    fields = {
        "block_ir_file": "File: The block IR file.",
        "delay_model": "Optional(string) Delay model used in codegen.",
        "module_sig_file": "File: The module signature of the Verilog file.",
        "schedule_file": "File: The schedule of the module.",
        "top": "String: Name of top level block in the IR.",
        "verilog_line_map_file": "File: The Verilog line map file.",
        "verilog_file": "File: The Verilog file.",
        "pipeline_stages": "Optional(string): The number of pipeline stages.",
        "clock_period_ps": "Optional(string): The clock period used for " +
                           "scheduling.",
    },
)

JitWrapperInfo = provider(
    doc = "A provider containing JIT Wrapper file information for the " +
          "target. It is created and returned by the xls_ir_jit_wrapper rule.",
    fields = {
        "header_file": "File: The header file.",
        "source_file": "File: The source file.",
    },
)
