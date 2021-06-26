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
This module contains build macros for XLS.
"""

load(
    "//xls/build_rules:xls_config_rules.bzl",
    "enable_generated_file_wrapper",
)
load(
    "//xls/build_rules:xls_rules.bzl",
    "get_xls_dslx_verilog_generated_files",
    "xls_dslx_verilog",
)

def xls_dslx_verilog_macro(
        name,
        dep,
        ir_conv_args = {},
        opt_ir_args = {},
        codegen_args = {},
        enable_generated_file = True,
        enable_presubmit_generated_file = False,
        **kwargs):
    """A macro wrapper for the 'xls_dslx_verilog' rule.

    The macro instantiates the 'xls_dslx_verilog' rule and
    'enable_generated_file_wrapper' function. The generated files of the rule
    are listed in the outs attribute of the rule.

    Args:
      name: The name of the rule.
      dep: The 'xls_dslx_module_library' target used for dependency. See 'dep'
        attribute from the 'xls_dslx_verilog' rule.
      ir_conv_args: IR conversion Arguments. See 'ir_conv_args' attribute from
        the 'xls_dslx_ir' rule.
      opt_ir_args: IR optimization Arguments. See 'opt_ir_args' attribute from
        the 'xls_ir_opt_ir' rule.
      codegen_args: Codegen Arguments. See 'codegen_args' attribute from the
        'xls_ir_verilog' rule.
      enable_generated_file: See 'enable_generated_file' from
        'enable_generated_file_wrapper' function.
      enable_presubmit_generated_file: See 'enable_presubmit_generated_file'
        from 'enable_generated_file_wrapper' function.
      **kwargs: Positional arguments. Named arguments.
    """

    # Type check input
    if type(name) != type(""):
        fail("Argument 'name' must be of string type.")
    if type(dep) != type(""):
        fail("Argument 'dep' must be of string type.")
    if type(ir_conv_args) != type({}):
        fail("Argument 'ir_conv_args' must be of dictionary type.")
    if type(opt_ir_args) != type({}):
        fail("Argument 'opt_ir_args' must be of dictionary type.")
    if type(codegen_args) != type({}):
        fail("Argument 'codegen_args' must be of dictionary type.")
    if type(enable_generated_file) != type(True):
        fail("Argument 'enable_generated_file' must be of boolean type.")
    if type(enable_presubmit_generated_file) != type(True):
        fail("Argument 'enable_presubmit_generated_file' must be " +
             "of boolean type.")

    xls_dslx_verilog(
        name = name,
        dep = dep,
        ir_conv_args = ir_conv_args,
        opt_ir_args = opt_ir_args,
        codegen_args = codegen_args,
        outs = get_xls_dslx_verilog_generated_files(name, codegen_args),
        **kwargs
    )
    enable_generated_file_wrapper(
        wrapped_target = name,
        tags = kwargs.get("tags", None),
        testonly = kwargs.get("testonly", None),
        enable_generated_file = enable_generated_file,
        enable_presubmit_generated_file = enable_presubmit_generated_file,
    )
