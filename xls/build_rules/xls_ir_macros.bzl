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
This module contains IR-related build macros for XLS.
"""

load(
    "//xls/build_rules:xls_config_rules.bzl",
    "enable_generated_file_wrapper",
)
load(
    "//xls/build_rules:xls_internal_aot_rules.bzl",
    "xls_aot_generate",
)
load(
    "//xls/build_rules:xls_internal_build_defs.bzl",
    "XLS_IS_MSAN_BUILD",
)
load(
    "//xls/build_rules:xls_ir_rules.bzl",
    "append_xls_dslx_ir_generated_files",
    "append_xls_ir_opt_ir_generated_files",
    "get_xls_dslx_ir_generated_files",
    "get_xls_ir_opt_ir_generated_files",
    "xls_dslx_ir",
    "xls_ir_cc_library",
    "xls_ir_opt_ir",
)
load(
    "//xls/build_rules:xls_type_check_utils.bzl",
    "bool_type_check",
    "dictionary_type_check",
    "list_type_check",
    "string_type_check",
)

def xls_dslx_ir_macro(
        name,
        dslx_top,
        srcs = None,
        deps = None,
        library = None,
        ir_conv_args = {},
        enable_generated_file = True,
        enable_presubmit_generated_file = False,
        **kwargs):
    """A macro that instantiates a build rule converting a DSLX source file to an IR file.

    The macro instantiates a rule that converts a DSLX source file to an IR
    file. The macro also instantiates the 'enable_generated_file_wrapper'
    function. The generated files are listed in the outs attribute of the rule.

    Example:

    An IR conversion with a top entity defined.

        ```
        # Assume a xls_dslx_library target bc_dslx is present.
        xls_dslx_ir(
            name = "d_ir",
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
        //xls/dslx/ir_convert/ir_converter_main.cc. Note: the 'top'
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
    bool_type_check("enable_generated_file", enable_generated_file)
    bool_type_check("enable_presubmit_generated_file", enable_presubmit_generated_file)

    # Append output files to arguments.
    kwargs = append_xls_dslx_ir_generated_files(kwargs, name)

    xls_dslx_ir(
        name = name,
        srcs = srcs,
        deps = deps,
        library = library,
        dslx_top = dslx_top,
        ir_conv_args = ir_conv_args,
        outs = get_xls_dslx_ir_generated_files(kwargs),
        **kwargs
    )
    enable_generated_file_wrapper(
        wrapped_target = name,
        enable_generated_file = enable_generated_file,
        enable_presubmit_generated_file = enable_presubmit_generated_file,
        **kwargs
    )

def xls_ir_opt_ir_macro(
        name,
        src,
        opt_ir_args = {},
        enable_generated_file = True,
        enable_presubmit_generated_file = False,
        debug_srcs = [],
        **kwargs):
    """A macro that instantiates a build rule optimizing an IR file.

    The macro instantiates a build rule that optimizes an IR file. The macro
    also instantiates the 'enable_generated_file_wrapper' function. The
    generated files are listed in the outs attribute of the rule.

    Examples:

    1.  A simple example.

        ```
        xls_ir_opt_ir(
            name = "a_opt_ir",
            src = "a.ir",
        )
        ```

    1.  Optimizing an IR file with a top entity defined.

        ```
        xls_ir_opt_ir(
            name = "a_opt_ir",
            src = "a.ir",
            opt_ir_args = {
                "inline_procs" : "true",
            },
        )
        ```

    Args:
      name: The name of the rule.
      src: The IR source file. A single source file must be provided. The file
        must have a '.ir' extension.
      opt_ir_args: Arguments of the IR optimizer tool. For details on the
        arguments, refer to the opt_main application at
        //xls/tools/opt_main.cc. Note: the 'top'
        argument is not assigned using this attribute.
      enable_generated_file: See 'enable_generated_file' from
        'enable_generated_file_wrapper' function.
      enable_presubmit_generated_file: See 'enable_presubmit_generated_file'
        from 'enable_generated_file_wrapper' function.
      debug_srcs: List of additional source files for debugging info. Allows opt_main to correctly
        display lines from original source file (e.g. the .cc file before the xlscc pass) when an
        error occurs.
      **kwargs: Keyword arguments. Named arguments.
    """

    # Type check input
    string_type_check("name", name)
    string_type_check("src", src)
    dictionary_type_check("opt_ir_args", opt_ir_args)
    bool_type_check("enable_generated_file", enable_generated_file)
    bool_type_check("enable_presubmit_generated_file", enable_presubmit_generated_file)

    # Append output files to arguments.
    kwargs = append_xls_ir_opt_ir_generated_files(kwargs, name)

    xls_ir_opt_ir(
        name = name,
        src = src,
        opt_ir_args = opt_ir_args,
        outs = get_xls_ir_opt_ir_generated_files(kwargs),
        debug_srcs = debug_srcs,
        **kwargs
    )
    enable_generated_file_wrapper(
        wrapped_target = name,
        enable_generated_file = enable_generated_file,
        enable_presubmit_generated_file = enable_presubmit_generated_file,
        **kwargs
    )

def xls_ir_cc_library_macro(
        name,
        src,
        top = None,
        namespaces = ""):
    """Invokes the AOT compiles the input IR into a cc_library.

    Example:

    ```
    xls_ir_opt_ir(
        name "foo",
        ...
    )

    xls_ir_cc_library_macro(
        name = "foo_cc",
        src = ":foo.opt.ir",
        top = "bar",
        namespaces = "a,b,c",
    )
    ```

    This will produce a cc_library that will execute the fn `bar` from the
    `foo` IR file. The call itself will be inside the namespace `a::b::c`.

    Args:
      name: The name of the resulting library.
      src: The path to the IR file to compile.
      top: The entry point in the IR file of interest.
      namespaces: A comma-separated list of namespaces into which the
                  generated code should go.
    """
    string_type_check("name", name)
    string_type_check("src", src)
    string_type_check("top", top, True)
    string_type_check("namespaces", namespaces)

    aot_name = name + "_gen_aot"
    xls_aot_generate(
        name = aot_name,
        src = src,
        top = top,
        # The XLS AOT compiler does not currently support cross-compilation.
        with_msan = XLS_IS_MSAN_BUILD,
    )

    wrapper_name = name + "_gen_aot_wrapper"
    xls_ir_cc_library(
        name = wrapper_name,
        file_basename = name,
        aot_info = ":" + aot_name,
        src = src,
        namespaces = namespaces,
    )

    native.cc_library(
        name = name,
        srcs = [
            ":" + wrapper_name,
        ],
        hdrs = [
            ":" + wrapper_name,
        ],
        # The XLS AOT compiler does not currently support cross-compilation.
        deps = [
            ":" + aot_name,
            "@com_google_absl//absl/status:statusor",
            "@com_google_absl//absl/types:span",
            "//xls/ir:events",
            "//xls/ir:value",
            "//xls/jit:aot_runtime",
            "//xls/jit:type_layout",
        ],
    )
