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
This module contains jit-wrapper-related build rules for XLS.
"""

load("@bazel_skylib//lib:dicts.bzl", "dicts")
load(
    "//xls/build_rules:xls_common_rules.bzl",
    "get_runfiles_for_xls",
    "get_src_ir_for_xls",
    "split_filename",
)
load(
    "//xls/build_rules:xls_config_rules.bzl",
    "CONFIG",
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
    "xls_ir_common_attrs",
    "xls_ir_top_attrs",
)
load("//xls/build_rules:xls_providers.bzl", "AotCompileInfo", "JitWrapperInfo")
load(
    "//xls/build_rules:xls_toolchains.bzl",
    "xls_toolchain_attrs",
)
load(
    "//xls/build_rules:xls_type_check_utils.bzl",
    "bool_type_check",
    "dictionary_type_check",
    "string_type_check",
)

_H_FILE_EXTENSION = ".h"

_CC_FILE_EXTENSION = ".cc"

_xls_ir_jit_wrapper_attrs = {
    "jit_wrapper_args": attr.string_dict(
        doc = "Arguments of the JIT wrapper tool.",
    ),
    "source_file": attr.output(
        doc = "The filename of the generated source file. The filename must " +
              "have a '" + _CC_FILE_EXTENSION + "' extension.",
        mandatory = True,
    ),
    "header_file": attr.output(
        doc = "The filename of the generated header file. The filename must " +
              "have a '" + _H_FILE_EXTENSION + "' extension.",
        mandatory = True,
    ),
    "wrapper_type": attr.string(
        doc = "type of function_base we are wrapping.",
        mandatory = True,
    ),
    "aot_info": attr.label(
        doc = "The target which contains information about available AOT code.",
        mandatory = True,
    ),
}

def _xls_ir_jit_wrapper_impl(ctx):
    """The implementation of the 'xls_ir_jit_wrapper' rule.

    Execute the JIT wrapper tool on the IR file.

    Args:
      ctx: The current rule's context object.

    Returns:
      JitWrapperInfo provider
      DefaultInfo provider
    """
    jit_wrapper_tool = ctx.executable._xls_jit_wrapper_tool

    # default arguments
    jit_wrapper_args = ctx.attr.jit_wrapper_args
    jit_wrapper_flags = ctx.actions.args()

    # parse arguments
    JIT_WRAPPER_FLAGS = (
        "class_name",
        "function",
        "namespace",
    )

    if "namespace" not in jit_wrapper_args:
        fail("Must specify 'namespace' in jit_wrapper_args")

    for flag_name in jit_wrapper_args:
        if flag_name in JIT_WRAPPER_FLAGS:
            if flag_name == "namespace":
                # `namespace` is a C++ keyword which prevents its use as a flag
                # name.
                jit_wrapper_flags.add(
                    "--wrapper_namespace",
                    jit_wrapper_args[flag_name],
                )
            else:
                jit_wrapper_flags.add(
                    "--{}".format(flag_name),
                    jit_wrapper_args[flag_name],
                )
        else:
            fail("Unrecognized argument: %s." % flag_name)

    # source file
    src = get_src_ir_for_xls(ctx)
    jit_wrapper_flags.add("--ir_path", src.ir_file.path)

    # Retrieve basename and extension from filename
    source_filename = ctx.outputs.source_file.basename
    header_filename = ctx.outputs.header_file.basename
    source_basename, source_extension = split_filename(source_filename)
    header_basename, header_extension = split_filename(header_filename)

    # validate filename extension
    if source_extension != _CC_FILE_EXTENSION[1:]:
        fail("Source filename must contain the '%s' extension." %
             _CC_FILE_EXTENSION)
    if header_extension != _H_FILE_EXTENSION[1:]:
        fail("Header filename must contain the '%s' extension." %
             _H_FILE_EXTENSION)

    if ctx.attr.top != "":
        jit_wrapper_flags.add("--function", ctx.attr.top)

    # validate basename
    if source_basename != header_basename:
        fail("The basename of the source and header files do not match.")

    # Append to argument list.
    jit_wrapper_flags.add("--output_name", source_basename)

    cc_file = ctx.actions.declare_file(source_filename)
    h_file = ctx.actions.declare_file(header_filename)

    # output directory
    jit_wrapper_flags.add("--output_dir", cc_file.dirname)

    # genfiles directory
    jit_wrapper_flags.add("--genfiles_dir", ctx.bin_dir.path + "/")

    # function_type
    jit_wrapper_flags.add("--function_type", ctx.attr.wrapper_type)

    # Aot information
    aot_info_file = ctx.attr.aot_info[AotCompileInfo].proto_file
    jit_wrapper_flags.add("--aot_info", aot_info_file.path)

    my_generated_files = [cc_file, h_file]

    # Get runfiles
    jit_wrapper_tool_runfiles = ctx.attr._xls_jit_wrapper_tool[DefaultInfo].default_runfiles
    runfiles = get_runfiles_for_xls(ctx, [jit_wrapper_tool_runfiles], [src.ir_file, aot_info_file])

    ctx.actions.run(
        outputs = my_generated_files,
        tools = [jit_wrapper_tool],
        inputs = runfiles.files,
        arguments = [jit_wrapper_flags],
        executable = jit_wrapper_tool.path,
        mnemonic = "IRJITWrapper",
        progress_message = "Building JIT wrapper for source file: %s" % (src.ir_file.path),
        toolchain = None,
    )
    return [
        JitWrapperInfo(
            source_file = cc_file,
            header_file = h_file,
        ),
        DefaultInfo(
            files = depset(my_generated_files),
            runfiles = runfiles,
        ),
    ]

xls_ir_jit_wrapper = rule(
    doc = """A build rule that generates the sources for JIT invocation wrappers.

Examples:

1. A file as the source.

    ```
    xls_ir_jit_wrapper(
        name = "a_jit_wrapper",
        src = "a.ir",
    )
    ```

1. An xls_ir_opt_ir target as the source.

    ```
    xls_ir_opt_ir(
        name = "a",
        src = "a.ir",
    )

    xls_ir_jit_wrapper(
        name = "a_jit_wrapper",
        src = ":a",
    )
    ```
    """,
    implementation = _xls_ir_jit_wrapper_impl,
    attrs = dicts.add(
        xls_ir_common_attrs,
        xls_ir_top_attrs,
        _xls_ir_jit_wrapper_attrs,
        CONFIG["xls_outs_attrs"],
        xls_toolchain_attrs,
    ),
)

def _no_aot_info_impl(ctx):
    """Helper rule to create an empty AotInfo proto."""
    file = ctx.actions.declare_file(ctx.attr.name + ".pb")
    ctx.actions.write(file, "", is_executable = False)
    return [
        DefaultInfo(files = depset([file])),
        AotCompileInfo(object_file = None, proto_file = file),
    ]

_no_aot_info = rule(
    doc = """Internal only utility rule to generate an empty AotCompileInfo proto file.

    This can be used with function types that don't yet support AOT.
    """,
    implementation = _no_aot_info_impl,
    attrs = {},
)

def xls_ir_jit_wrapper_macro(
        name,
        src,
        top,
        source_file,
        header_file,
        wrapper_type,
        aot_info,
        jit_wrapper_args = {},
        enable_generated_file = True,
        enable_presubmit_generated_file = False,
        **kwargs):
    """A macro wrapper for the 'xls_ir_jit_wrapper' rule.

    The macro instantiates the 'xls_ir_jit_wrapper' rule and
    'enable_generated_file_wrapper' function. The generated files of the rule
    are listed in the outs attribute of the rule.

    Args:
      name: The name of the rule.
      src: The IR file. See 'src' attribute from the 'xls_ir_jit_wrapper' rule.
      top: Name of the function/block/proc to wrap.
      source_file: The generated source file. See 'source_file' attribute from
        the 'xls_ir_jit_wrapper' rule.
      header_file: The generated header file. See 'header_file' attribute from
        the 'xls_ir_jit_wrapper' rule.
      wrapper_type: What sort of function base are we wrapping.
      aot_info: AotCompileInfo generating label with information about the AOT
        code that is available.
      jit_wrapper_args: Arguments of the JIT tool. See 'jit_wrapper_args'
         attribute from the 'xls_ir_jit_wrapper' rule.
      enable_generated_file: See 'enable_generated_file' from
        'enable_generated_file_wrapper' function.
      enable_presubmit_generated_file: See 'enable_presubmit_generated_file'
        from 'enable_generated_file_wrapper' function.
      **kwargs: Keyword arguments. Named arguments.
    """

    # Type check input
    string_type_check("name", name)
    string_type_check("src", src)
    string_type_check("top", top)
    string_type_check("source_file", source_file)
    string_type_check("header_file", header_file)
    string_type_check("wrapper_type", wrapper_type)
    string_type_check("aot_info", aot_info)
    dictionary_type_check("jit_wrapper_args", jit_wrapper_args)
    bool_type_check("enable_generated_file", enable_generated_file)
    bool_type_check("enable_presubmit_generated_file", enable_presubmit_generated_file)

    xls_ir_jit_wrapper(
        name = name,
        src = src,
        top = top,
        source_file = source_file,
        header_file = header_file,
        aot_info = aot_info,
        jit_wrapper_args = jit_wrapper_args,
        wrapper_type = wrapper_type,
        outs = [source_file, header_file],
        **kwargs
    )

    enable_generated_file_wrapper(
        wrapped_target = name,
        enable_generated_file = enable_generated_file,
        enable_presubmit_generated_file = enable_presubmit_generated_file,
        **kwargs
    )

FUNCTION_WRAPPER_TYPE = "FUNCTION"
PROC_WRAPPER_TYPE = "PROC"
BLOCK_WRAPPER_TYPE = "BLOCK"

_BASE_JIT_WRAPPER_DEPS = {
    FUNCTION_WRAPPER_TYPE: ["//xls/jit:function_base_jit_wrapper"],
    PROC_WRAPPER_TYPE: [
        "//xls/jit:proc_base_jit_wrapper",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
    BLOCK_WRAPPER_TYPE: [
        "//xls/jit:block_base_jit_wrapper",
        "//xls/ir:bits",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
}

def cc_xls_ir_jit_wrapper(
        name,
        src,
        jit_wrapper_args = {},
        wrapper_type = FUNCTION_WRAPPER_TYPE,
        top = "",
        llvm_opt_level = 3,
        **kwargs):
    """Invokes the JIT wrapper generator and compiles the result as a cc_library.

    The macro invokes the JIT wrapper generator on an IR source file. The
    generated source files are the inputs to a cc_library with its target name
    identical to this macro.

    Args:
      name: The name of the cc_library target.
      src: The path to the IR file.
      jit_wrapper_args: Arguments of the JIT wrapper tool. Note: argument
                        'output_name' cannot be defined.
      wrapper_type: The type of XLS construct to wrap. Must be one of 'BLOCK',
                    'FUNCTION', or 'PROC'. You should use the exported
                    FUNCTION_WRAPPER_TYPE, BLOCK_WRAPPER_TYPE, or
                    PROC_WRAPPER_TYPE symbols. Defaults to FUNCTION_WRAPPER_TYPE
                    for compatibility.
      llvm_opt_level: what opt level to configure aot compiled code to use.
      top: Name of the top function/proc/block.
      **kwargs: Keyword arguments. Named arguments.
    """

    dictionary_type_check("jit_wrapper_args", jit_wrapper_args)
    string_type_check("src", src)
    string_type_check("top", src)

    # Validate arguments of macro
    if kwargs.get("source_file"):
        fail("Cannot set 'source_file' attribute in macro '%s' of type " +
             "'cc_xls_ir_jit_wrapper'." % name)
    if kwargs.get("header_file"):
        fail("Cannot set 'header_file' attribute in macro '%s' of type " +
             "'cc_xls_ir_jit_wrapper'." % name)

    if wrapper_type not in (FUNCTION_WRAPPER_TYPE, BLOCK_WRAPPER_TYPE, PROC_WRAPPER_TYPE):
        fail(("Cannot set 'wrapper_type' to %s. It must be one of BLOCK_WRAPPER_TYPE, " +
              "FUNCTION_WRAPPER_TYPE, or PROC_WRAPPER_TYPE") % wrapper_type)

    source_filename = name + _CC_FILE_EXTENSION
    header_filename = name + _H_FILE_EXTENSION

    extra_lib_deps = []
    xls_aot_generate(
        name = name + "_aot_code_for_wrapper",
        src = src,
        top = top,
        with_msan = XLS_IS_MSAN_BUILD,
        llvm_opt_level = llvm_opt_level,
        # The XLS AOT compiler does not currently support cross-compilation.
    )
    aot_info_target = ":" + name + "_aot_code_for_wrapper"
    extra_lib_deps.append(aot_info_target)

    xls_ir_jit_wrapper_macro(
        name = "__" + name + "_xls_ir_jit_wrapper",
        src = src,
        top = top,
        jit_wrapper_args = jit_wrapper_args,
        aot_info = aot_info_target,
        wrapper_type = wrapper_type,
        source_file = source_filename,
        header_file = header_filename,
        **kwargs
    )

    native.cc_library(
        name = name,
        srcs = [":" + source_filename],
        hdrs = [":" + header_filename],
        deps = extra_lib_deps +
               _BASE_JIT_WRAPPER_DEPS[wrapper_type] + [
            "@com_google_absl//absl/status",
            "@com_google_absl//absl/status:statusor",
            "//xls/common/status:status_macros",
            "//xls/interpreter:evaluator_options",
            "//xls/public:value",
        ],
        **kwargs
    )
