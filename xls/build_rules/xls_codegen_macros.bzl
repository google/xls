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
This module contains codegen-related build macros for XLS.
"""

load(
    "//xls/build_rules:xls_codegen_rules.bzl",
    "append_xls_ir_verilog_generated_files",
    "get_xls_ir_verilog_generated_files",
    "validate_verilog_filename",
    "xls_ir_verilog",
)
load(
    "//xls/build_rules:xls_common_rules.bzl",
    "split_filename",
)
load(
    "//xls/build_rules:xls_config_rules.bzl",
    "enable_generated_file_wrapper",
)

def xls_ir_verilog_macro(
        name,
        src,
        verilog_file,
        codegen_args = {},
        enable_generated_file = True,
        enable_presubmit_generated_file = False,
        **kwargs):
    """A macro that instantiates a build rule generating a Verilog file from an IR file.

    The macro instantiates a build rule that generate a Verilog file from an IR
    file and the 'enable_generated_file_wrapper' function. The generated files
    are listed in the outs attribute of the rule.

    Example:

        ```
        xls_ir_verilog(
            name = "a_verilog",
            src = "a.ir",
            codegen_args = {
                "pipeline_stages": "1",
                ...
            },
        )
        ```

    Args:
      name: The name of the rule.
      src: The IR source file. A single source file must be provided. The file
        must have a '.ir' extension.
      codegen_args: Arguments of the codegen tool. For details on the arguments,
        refer to the codegen_main application at
        //xls/tools/codegen_main.cc.
      verilog_file: The filename of Verilog file generated. The filename must
        have a '.v' or '.sv', extension.
      enable_generated_file: See 'enable_generated_file' from
        'enable_generated_file_wrapper' function.
      enable_presubmit_generated_file: See 'enable_presubmit_generated_file'
        from 'enable_generated_file_wrapper' function.
      **kwargs: Keyword arguments. Named arguments.
    """

    # Type check input
    if type(name) != type(""):
        fail("Argument 'name' must be of string type.")
    if type(src) != type(""):
        fail("Argument 'src' must be of string type.")
    if type(verilog_file) != type(""):
        fail("Argument 'verilog_file' must be of string type.")
    if type(codegen_args) != type({}):
        fail("Argument 'codegen_args' must be of dictionary type.")
    if type(enable_generated_file) != type(True):
        fail("Argument 'enable_generated_file' must be of boolean type.")
    if type(enable_presubmit_generated_file) != type(True):
        fail("Argument 'enable_presubmit_generated_file' must be " +
             "of boolean type.")

    # Append output files to arguments.
    use_system_verilog = ("use_system_verilog" in codegen_args and
                          codegen_args["use_system_verilog"].lower() == "true")
    validate_verilog_filename(verilog_file, use_system_verilog)
    verilog_basename = split_filename(verilog_file)[0]
    kwargs = append_xls_ir_verilog_generated_files(
        kwargs,
        verilog_basename,
        codegen_args,
    )

    xls_ir_verilog(
        name = name,
        src = src,
        codegen_args = codegen_args,
        verilog_file = verilog_file,
        outs = get_xls_ir_verilog_generated_files(kwargs, codegen_args) +
               [verilog_file],
        **kwargs
    )
    enable_generated_file_wrapper(
        wrapped_target = name,
        enable_generated_file = enable_generated_file,
        enable_presubmit_generated_file = enable_presubmit_generated_file,
        **kwargs
    )
