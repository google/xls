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

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load(
    "//xls/build_rules:xls_codegen_rules.bzl",
    "append_xls_ir_verilog_generated_files",
    "get_xls_ir_verilog_generated_files",
    "validate_verilog_filename",
)
load(
    "//xls/build_rules:xls_config_rules.bzl",
    "enable_generated_file_wrapper",
)
load(
    "//xls/build_rules:xls_common_rules.bzl",
    "split_filename",
)
load(
    "//xls/build_rules:xls_ir_rules.bzl",
    "append_xls_dslx_ir_generated_files",
    "append_xls_ir_opt_ir_generated_files",
    "get_xls_dslx_ir_generated_files",
    "get_xls_ir_opt_ir_generated_files",
)
load(
    "//xls/build_rules:xls_rules.bzl",
    "xls_dslx_opt_ir",
    "xls_dslx_verilog",
)
load(
    "//xls/build_rules:xls_type_check_helpers.bzl",
    "bool_type_check",
    "dictionary_type_check",
    "list_type_check",
    "string_type_check",
)

def _xls_dslx_verilog_macro(
        name,
        dslx_top,
        verilog_file,
        srcs = None,
        deps = None,
        library = None,
        ir_conv_args = {},
        opt_ir_args = {},
        codegen_args = {},
        enable_generated_file = True,
        enable_presubmit_generated_file = False,
        **kwargs):
    """A macro that instantiates a build rule generating a Verilog file from a DSLX source file.

    The macro instantiates a build rule that generates a Verilog file from a DSLX source file. The
    build rule executes the core functionality of following macros:

    1. xls_dslx_ir (converts a DSLX file to an IR),
    1. xls_ir_opt_ir (optimizes the IR), and,
    1. xls_ir_verilog (generated a Verilog file).

    The macro also instantiates the 'enable_generated_file_wrapper'
    function. The generated files are listed in the outs attribute of the rule.

    This macro is used by the 'xls_dslx_verilog_build_and_test' macro.

    Examples:

    1. A simple example.

        ```
        # Assume a xls_dslx_library target bc_dslx is present.
        xls_dslx_verilog(
            name = "d_verilog",
            srcs = ["d.x"],
            deps = [":bc_dslx"],
            codegen_args = {
                "pipeline_stages": "1",
            },
            dslx_top = "d",
        )
        ```

    Args:
      name: The name of the rule.
      srcs: Top level source files for the conversion. Files must have a '.x'
        extension. There must be single source file.
      deps: Dependency targets for the files in the 'srcs' argument.
      library: A DSLX library target where the direct (non-transitive)
        files of the target are tested. This argument is mutually
        exclusive with the 'srcs' and 'deps' arguments.
      verilog_file: The filename of Verilog file generated. The filename must
        have a '.v' extension.
      dslx_top: The top entity to perform the IR conversion.
      ir_conv_args: Arguments of the IR conversion tool. For details on the
        arguments, refer to the ir_converter_main application at
        //xls/dslx/ir_converter_main.cc. Note: the 'top'
        argument is not assigned using this attribute.
      opt_ir_args: Arguments of the IR optimizer tool. For details on the
        arguments, refer to the opt_main application at
        //xls/tools/opt_main.cc. Note: the 'top'
        argument is not assigned using this attribute.
      codegen_args: Arguments of the codegen tool. For details on the arguments,
        refer to the codegen_main application at
        //xls/tools/codegen_main.cc.
      enable_generated_file: See 'enable_generated_file' from
        'enable_generated_file_wrapper' function.
      enable_presubmit_generated_file: See 'enable_presubmit_generated_file'
        from 'enable_generated_file_wrapper' function.
      **kwargs: Keyword arguments. Named arguments.
    """

    # Type check input
    string_type_check("name", name)
    list_type_check("srcs", srcs, True)
    list_type_check("deps", deps, True)
    string_type_check("library", library, True)
    string_type_check("dslx_top", dslx_top)
    string_type_check("verilog_file", verilog_file)
    dictionary_type_check("ir_conv_args", ir_conv_args)
    dictionary_type_check("opt_ir_args", opt_ir_args)
    dictionary_type_check("codegen_args", codegen_args)
    bool_type_check("enable_generated_file", enable_generated_file)
    bool_type_check("enable_presubmit_generated_file", enable_presubmit_generated_file)

    # Append output files to arguments.
    kwargs = append_xls_dslx_ir_generated_files(kwargs, name)
    kwargs = append_xls_ir_opt_ir_generated_files(kwargs, name)
    use_system_verilog = codegen_args.get("use_system_verilog", "True").lower() == "true"
    validate_verilog_filename(verilog_file, use_system_verilog)
    verilog_basename = split_filename(verilog_file)[0]
    kwargs = append_xls_ir_verilog_generated_files(
        kwargs,
        verilog_basename,
        codegen_args,
    )

    xls_dslx_verilog(
        name = name,
        srcs = srcs,
        deps = deps,
        library = library,
        dslx_top = dslx_top,
        verilog_file = verilog_file,
        ir_conv_args = ir_conv_args,
        opt_ir_args = opt_ir_args,
        codegen_args = codegen_args,
        outs = get_xls_dslx_ir_generated_files(kwargs) +
               get_xls_ir_opt_ir_generated_files(kwargs) +
               get_xls_ir_verilog_generated_files(kwargs, codegen_args) +
               [verilog_file],
        **kwargs
    )
    enable_generated_file_wrapper(
        wrapped_target = name,
        enable_generated_file = enable_generated_file,
        enable_presubmit_generated_file = enable_presubmit_generated_file,
        **kwargs
    )

def xls_dslx_verilog_build_and_test(
        name,
        dslx_top,
        verilog_file,
        srcs = None,
        deps = None,
        library = None,
        ir_conv_args = {},
        opt_ir_args = {},
        codegen_args = {},
        enable_generated_file = True,
        enable_presubmit_generated_file = False,
        **kwargs):
    """A macro that instantiates a build rule generating a Verilog file from a DSLX source file and tests the build.

    The macro instantiates a build rule that generates a Verilog file from a DSLX source file. The
    build rule executes the core functionality of following macros:

    1. xls_dslx_ir (converts a DSLX file to an IR),
    1. xls_ir_opt_ir (optimizes the IR), and,
    1. xls_ir_verilog (generated a Verilog file).

    The macro also instantiates a 'build_test' testing that the build rule generating a Verilog
    file. If the build is not successful, an error is produced when executing a test command on the
    target.

    Examples:

    1. A simple example.

        ```
        # Assume a xls_dslx_library target bc_dslx is present.
        xls_dslx_verilog(
            name = "d_verilog",
            srcs = ["d.x"],
            deps = [":bc_dslx"],
            codegen_args = {
                "pipeline_stages": "1",
            },
            dslx_top = "d",
        )
        ```

    Args:
      name: The name of the rule.
      srcs: Top level source files for the conversion. Files must have a '.x'
        extension. There must be single source file.
      deps: Dependency targets for the files in the 'srcs' argument.
      library: A DSLX library target where the direct (non-transitive)
        files of the target are tested. This argument is mutually
        exclusive with the 'srcs' and 'deps' arguments.
      verilog_file: The filename of Verilog file generated. The filename must
        have a '.v' extension.
      dslx_top: The top entity to perform the IR conversion.
      ir_conv_args: Arguments of the IR conversion tool. For details on the
        arguments, refer to the ir_converter_main application at
        //xls/dslx/ir_converter_main.cc. Note: the 'top'
        argument is not assigned using this attribute.
      opt_ir_args: Arguments of the IR optimizer tool. For details on the
        arguments, refer to the opt_main application at
        //xls/tools/opt_main.cc. Note: the 'top'
        argument is not assigned using this attribute.
      codegen_args: Arguments of the codegen tool. For details on the arguments,
        refer to the codegen_main application at
        //xls/tools/codegen_main.cc.
      enable_generated_file: See 'enable_generated_file' from
        'enable_generated_file_wrapper' function.
      enable_presubmit_generated_file: See 'enable_presubmit_generated_file'
        from 'enable_generated_file_wrapper' function.
      **kwargs: Keyword arguments. Named arguments.
    """
    _xls_dslx_verilog_macro(
        name = name,
        dslx_top = dslx_top,
        verilog_file = verilog_file,
        srcs = srcs,
        deps = deps,
        library = library,
        ir_conv_args = ir_conv_args,
        opt_ir_args = opt_ir_args,
        codegen_args = codegen_args,
        enable_generated_file = enable_generated_file,
        enable_presubmit_generated_file = enable_presubmit_generated_file,
        **kwargs
    )
    build_test(
        name = "__" + name,
        targets = [":" + name],
    )

def xls_dslx_opt_ir_macro(
        name,
        dslx_top,
        srcs = None,
        deps = None,
        library = None,
        ir_conv_args = {},
        opt_ir_args = {},
        enable_generated_file = True,
        enable_presubmit_generated_file = False,
        **kwargs):
    """A macro that instantiates a build rule generating an optimized IR file from a DSLX source file.

    The macro instantiates a build rule that generates an optimized IR file from
    a DSLX source file. The build rule executes the core functionality of
    following macros:

    1. xls_dslx_ir (converts a DSLX file to an IR), and,
    1. xls_ir_opt_ir (optimizes the IR).

    The macro also instantiates the 'enable_generated_file_wrapper'
    function. The generated files are listed in the outs attribute of the rule.

    Examples:

    1. A simple example.

        ```
        # Assume a xls_dslx_library target bc_dslx is present.
        xls_dslx_opt_ir(
            name = "d_opt_ir",
            srcs = ["d.x"],
            deps = [":bc_dslx"],
            dslx_top = "d",
        )
        ```

    Args:
      name: The name of the rule.
      srcs: Top level source files for the conversion. Files must have a '.x'
        extension. There must be single source file.
      deps: Dependency targets for the files in the 'srcs' argument.
      library: A DSLX library target where the direct (non-transitive)
        files of the target are tested. This argument is mutually
        exclusive with the 'srcs' and 'deps' arguments.
      dslx_top: The top entity to perform the IR conversion.
      ir_conv_args: Arguments of the IR conversion tool. For details on the
        arguments, refer to the ir_converter_main application at
        //xls/dslx/ir_converter_main.cc. Note: the 'top'
        argument is not assigned using this attribute.
      opt_ir_args: Arguments of the IR optimizer tool. For details on the
        arguments, refer to the opt_main application at
        //xls/tools/opt_main.cc. Note: the 'top'
        argument is not assigned using this attribute.
      enable_generated_file: See 'enable_generated_file' from
        'enable_generated_file_wrapper' function.
      enable_presubmit_generated_file: See 'enable_presubmit_generated_file'
        from 'enable_generated_file_wrapper' function.
      **kwargs: Keyword arguments. Named arguments.
    """

    # Type check input
    string_type_check("name", name)
    list_type_check("srcs", srcs, True)
    list_type_check("deps", deps, True)
    string_type_check("library", library, True)
    string_type_check("dslx_top", dslx_top)
    dictionary_type_check("ir_conv_args", ir_conv_args)
    dictionary_type_check("opt_ir_args", opt_ir_args)
    bool_type_check("enable_generated_file", enable_generated_file)
    bool_type_check("enable_presubmit_generated_file", enable_presubmit_generated_file)

    # Append output files to arguments.
    kwargs = append_xls_dslx_ir_generated_files(kwargs, name)
    kwargs = append_xls_ir_opt_ir_generated_files(kwargs, name)

    xls_dslx_opt_ir(
        name = name,
        srcs = srcs,
        deps = deps,
        library = library,
        dslx_top = dslx_top,
        ir_conv_args = ir_conv_args,
        opt_ir_args = opt_ir_args,
        outs = get_xls_dslx_ir_generated_files(kwargs) +
               get_xls_ir_opt_ir_generated_files(kwargs),
        **kwargs
    )
    enable_generated_file_wrapper(
        wrapped_target = name,
        enable_generated_file = enable_generated_file,
        enable_presubmit_generated_file = enable_presubmit_generated_file,
        **kwargs
    )

def xls_dslx_cpp_type_library(
        name,
        src):
    """Creates a cc_library target for transpiled DSLX types.

    This macros invokes the DSLX-to-C++ transpiler and compiles the result as
    a cc_library with its target name identical to this macro.

    Args:
      name: The name of the eventual cc_library.
      src: The DSLX file whose types to compile as C++.
    """
    native.genrule(
        name = name + "_generate_sources",
        srcs = [src],
        outs = [
            name + ".h",
            name + ".cc",
        ],
        tools = [
            "//xls/dslx:cpp_transpiler_main",
        ],
        cmd = "$(location //xls/dslx:cpp_transpiler_main) " +
              "--output_header_path=$(@D)/{}.h ".format(name) +
              "--output_source_path=$(@D)/{}.cc ".format(name) +
              "$(location {})".format(src),
    )

    native.cc_library(
        name = name,
        srcs = [":" + name + ".cc"],
        hdrs = [":" + name + ".h"],
        deps = [
            "@com_google_absl//absl/base:core_headers",
            "@com_google_absl//absl/status:status",
            "@com_google_absl//absl/status:statusor",
            "@com_google_absl//absl/types:span",
            "//xls/public:value",
        ],
    )
