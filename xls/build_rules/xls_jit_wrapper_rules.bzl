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
load("//xls/build_rules:xls_common_rules.bzl", "split_filename")
load("//xls/build_rules:xls_config_rules.bzl", "CONFIG")
load("//xls/build_rules:xls_providers.bzl", "JitWrapperInfo")
load("//xls/build_rules:xls_ir_rules.bzl", "xls_ir_common_attrs")
load(
    "//xls/build_rules:xls_toolchains.bzl",
    "get_xls_toolchain_info",
    "xls_toolchain_attr",
)

_xls_ir_jit_wrapper_attrs = {
    "jit_wrapper_args": attr.string_dict(
        doc = "Arguments of the JIT wrapper tool.",
    ),
    "source_file": attr.output(
        doc = "The filename of the generated source file. The filename must " +
              "have a '.cc' extension.",
        mandatory = True,
    ),
    "header_file": attr.output(
        doc = "The filename of the generated header file. The filename must " +
              "have a '.h' extension.",
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
    jit_wrapper_tool = get_xls_toolchain_info(ctx).jit_wrapper_tool

    # default arguments
    jit_wrapper_args = ctx.attr.jit_wrapper_args
    jit_wrapper_flags = ctx.actions.args()

    # parse arguments
    JIT_WRAPPER_FLAGS = (
        "class_name",
        "function",
    )
    for flag_name in jit_wrapper_args:
        if flag_name in JIT_WRAPPER_FLAGS:
            jit_wrapper_flags.add(
                "--{}".format(flag_name),
                jit_wrapper_args[flag_name],
            )
        else:
            fail("Unrecognized argument: %s." % flag_name)

    # source file
    src = ctx.file.src
    jit_wrapper_flags.add("--ir_path", src.path)

    # Retrieve basename and extension from filename
    source_filename = ctx.outputs.source_file.basename
    header_filename = ctx.outputs.header_file.basename
    source_basename, source_extension = split_filename(source_filename)
    header_basename, header_extension = split_filename(header_filename)

    # validate filename extension
    if source_extension != "cc":
        fail("Source filename must contain the '.cc' extension.")
    if header_extension != "h":
        fail("Header filename must contain the '.h' extension.")

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
    jit_wrapper_flags.add("--genfiles_dir", ctx.genfiles_dir.path)
    my_generated_files = [cc_file, h_file]
    ctx.actions.run(
        outputs = my_generated_files,
        tools = [jit_wrapper_tool],
        inputs = [src, jit_wrapper_tool],
        arguments = [jit_wrapper_flags],
        executable = jit_wrapper_tool.path,
        mnemonic = "IRJITWrapper",
        progress_message = "Building JIT wrapper for source file: %s" % (src.path),
    )
    return [
        JitWrapperInfo(
            source_file = cc_file,
            header_file = h_file,
        ),
        DefaultInfo(
            files = depset(my_generated_files),
        ),
    ]

#TODO(vmirian) 12-28-2021 Do not expose to user.
xls_ir_jit_wrapper = rule(
    doc = """
    A build rule that generates the sources for JIT invocation wrappers.

        Example:

         1) A file as the source.

        ```
            xls_ir_jit_wrapper(
                name = "a_jit_wrapper",
                src = "a.ir",
            )
        ```

        2) An xls_ir_opt_ir target as the source.

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
        _xls_ir_jit_wrapper_attrs,
        CONFIG["xls_outs_attrs"],
        xls_toolchain_attr,
    ),
)

def cc_xls_ir_jit_wrapper(
        name,
        src = None,
        jit_wrapper_args = None,
        **kwargs):
    """Instantiates xls_ir_jit_wrapper and a cc_library target with the files.

    The macro generates sources files (.cc and .h) using the
    xls_ir_jit_wrapper rule. The source files are the input to a cc_library
    target with the same name as this macro.

    Args:
      name: The name of the cc_library target.
      src: The path to the IR file.
      jit_wrapper_args: Arguments of the JIT wrapper tool. Note: argument
                        'output_name' cannot be defined.
      **kwargs: Keyword arguments. Named arguments.
    """
    if jit_wrapper_args != None and type(jit_wrapper_args) != type({}):
        fail("JIT Wrapper arguments must be a dictionary.")
    if src == None:
        fail("The source must be defined.")
    if type(src) != type(""):
        fail("The source must be a string.")

    # Validate arguments of macro
    if kwargs.get("source_file"):
        fail("Cannot set 'source_file' attribute in macro '%s' of type " +
             "'cc_xls_ir_jit_wrapper'." % name)
    if kwargs.get("header_file"):
        fail("Cannot set 'header_file' attribute in macro '%s' of type " +
             "'cc_xls_ir_jit_wrapper'." % name)

    source_filename = name + ".cc"
    header_filename = name + ".h"
    pkg_name = native.package_name() + "/"
    xls_ir_jit_wrapper(
        name = "__" + name + "_xls_ir_jit_wrapper",
        src = src,
        jit_wrapper_args = jit_wrapper_args,
        outs = [pkg_name + source_filename, pkg_name + header_filename],
        source_file = source_filename,
        header_file = header_filename,
        **kwargs
    )
    native.cc_library(
        name = name,
        srcs = [":" + source_filename],
        hdrs = [":" + header_filename],
        deps = [
            "@com_google_absl//absl/status",
            "//xls/common/status:status_macros",
            "@com_google_absl//absl/status:statusor",
            "//xls/ir",
            "//xls/ir:ir_parser",
            "//xls/public:function_builder",
            "//xls/public:value",
            "//xls/jit:ir_jit",
        ],
        **kwargs
    )
