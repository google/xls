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
    doc = "A provider containing DSLX file information for the target. It is " +
          "created and returned by the xls_dslx_library rule.",
    fields = {
        "dslx_source_files": "Depset: A depset containing the DSLX source " +
                             "(.x) files of the target and its dependencies.",
        "dslx_dummy_files": "Depset: A depset containing the DSLX generated " +
                            "dummy (.dummy) files of the target and its " +
                            "dependencies. A DSLX dummy file is generated " +
                            "when a DSLX file is successfully parsed and " +
                            "type checked. It is used to create a dependency " +
                            "between xls_dslx_library targets.",
    },
)

DslxModuleInfo = provider(
    doc = "A provider containing DSLX file information for the target. It is " +
          "created and returned by the xls_dslx_module_library rule.",
    fields = {
        "dslx_source_files": "List: A list containing the DSLX source " +
                             "(.x) files of its dependencies.",
        "dslx_dummy_files": "List: A list containing the DSLX generated " +
                            "dummy (.dummy) files of its dependencies. See " +
                            "DslxInfo for details on a dummy file.",
        "dslx_source_module_file": "File: A file containing the DSLX source " +
                                   "(.x) files of the target.",
        "dslx_dummy_module_file": "File: A file containing the DSLX " +
                                  "generated dummy (.dummy) files of the " +
                                  "target. See DslxInfo for details on a " +
                                  "dummy file.",
    },
)

ConvIRInfo = provider(
    doc = "A provider containing IR conversion file information for the " +
          "target. It is created and returned by the xls_dslx_ir rule.",
    fields = {
        "dslx_source_file": "File: The DSLX source file.",
        "conv_ir_file": "File: The IR file converted from a DSLX source.",
    },
)

OptIRInfo = provider(
    doc = "A provider containing IR optimization file information for the " +
          "target. It is created and returned by the xls_ir_opt_ir rule.",
    fields = {
        "input_ir_file": "File: The IR file input file.",
        "opt_ir_file": "File: The IR optimized file.",
        "opt_ir_args": "Dictionary: The arguments for the IR optimizer.",
    },
)

CodegenInfo = provider(
    doc = "A provider containing Codegen file information for the target. It " +
          "is created and returned by the xls_ir_verilog rule.",
    fields = {
        "verilog_file": "File: The Verilog file.",
        "module_sig_file": "File: The module signature of the Verilog file.",
        "schedule_file": "File: The schedule of the module.",
    },
)

JitWrapperInfo = provider(
    doc = "A provider containing JIT Wrapper file information for the " +
          "target. It is created and returned by the xls_ir_jit_wrapper rule.",
    fields = {
        "source_file": "File: The source file.",
        "header_file": "File: The header file.",
    },
)
