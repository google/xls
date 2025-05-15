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

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load(
    "//xls/build_rules:xls_codegen_rules.bzl",
    "append_xls_ir_verilog_generated_files",
    "get_xls_ir_verilog_generated_files",
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
load(
    "//xls/build_rules:xls_type_check_utils.bzl",
    "bool_type_check",
    "dictionary_type_check",
    "string_type_check",
)

def _xls_ir_verilog_macro(
        name,
        src,
        verilog_file,
        codegen_args = {},
        codegen_options_proto = None,
        scheduling_options_proto = None,
        enable_generated_file = True,
        enable_presubmit_generated_file = False,
        **kwargs):
    """A macro that instantiates a build rule generating a Verilog file from an IR file.

    The macro instantiates a build rule that generate a Verilog file from an IR
    file and the 'enable_generated_file_wrapper' function. The generated files
    are listed in the outs attribute of the rule.

    This macro is used by the 'xls_ir_verilog_build_and_test' macro.

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
      codegen_options_proto: Filename of a protobuf with arguments of the codegen tool.
        For details on the arguments, refer to the codegen_main application at
        //xls/tools/codegen_main.cc.
      scheduling_options_proto: Filename of a protobuf with scheduling options arguments
        of the codegen tool.
        For details on the arguments, refer to the codegen_main application at
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
    string_type_check("name", name)
    string_type_check("src", src)
    string_type_check("verilog_file", verilog_file)
    dictionary_type_check("codegen_args", codegen_args, True)
    bool_type_check("enable_generated_file", enable_generated_file)
    bool_type_check("enable_presubmit_generated_file", enable_presubmit_generated_file)

    final_codegen_args = dict(codegen_args)

    # Append output files to arguments.
    verilog_basename = split_filename(verilog_file)[0]
    kwargs = append_xls_ir_verilog_generated_files(
        kwargs,
        verilog_basename,
        final_codegen_args,
    )

    xls_ir_verilog(
        name = name,
        src = src,
        codegen_args = final_codegen_args,
        codegen_options_proto = codegen_options_proto,
        scheduling_options_proto = scheduling_options_proto,
        verilog_file = verilog_file,
        outs = get_xls_ir_verilog_generated_files(kwargs, final_codegen_args) +
               [verilog_file],
        **kwargs
    )
    enable_generated_file_wrapper(
        wrapped_target = name,
        enable_generated_file = enable_generated_file,
        enable_presubmit_generated_file = enable_presubmit_generated_file,
        **kwargs
    )

def xls_ir_verilog_build_and_test(
        name,
        src,
        verilog_file,
        codegen_args = {},
        codegen_options_proto = None,
        scheduling_options_proto = None,
        enable_generated_file = True,
        enable_presubmit_generated_file = False,
        **kwargs):
    """A macro that instantiates a build rule generating a Verilog file from an IR file and tests the build.

    The macro instantiates a build rule that generate a Verilog file from an IR file, and a
    'build_test' testing that the build rule generating a Verilog file. If the build is not
    successful, an error is produced when executing a test command on the target.

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
      codegen_options_proto: Filename of a protobuf with arguments of the codegen tool.
        For details on the arguments, refer to the codegen_main application at
        //xls/tools/codegen_main.cc.
      scheduling_options_proto: Filename of a protobuf with scheduling options arguments
        of the codegen tool.
        For details on the arguments, refer to the codegen_main application at
        //xls/tools/codegen_main.cc.
      verilog_file: The filename of Verilog file generated. The filename must
        have a '.v' or '.sv', extension.
      enable_generated_file: See 'enable_generated_file' from
        'enable_generated_file_wrapper' function.
      enable_presubmit_generated_file: See 'enable_presubmit_generated_file'
        from 'enable_generated_file_wrapper' function.
      **kwargs: Keyword arguments. Named arguments.
    """
    _xls_ir_verilog_macro(
        name = name,
        src = src,
        verilog_file = verilog_file,
        codegen_args = codegen_args,
        codegen_options_proto = codegen_options_proto,
        scheduling_options_proto = scheduling_options_proto,
        enable_generated_file = enable_generated_file,
        enable_presubmit_generated_file = enable_presubmit_generated_file,
        **kwargs
    )
    build_test(
        name = "__" + name,
        targets = [":" + name],
        tags = kwargs["tags"] if "tags" in kwargs else [],
    )
