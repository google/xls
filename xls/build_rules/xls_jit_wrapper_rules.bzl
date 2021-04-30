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

load("//xls/build_rules:xls_ir_rules.bzl", "ir_common_attrs")

IRToJitWrapperInfo = provider(
    doc = "A provider containing JIT Wrapper file information for the " +
          "target. It is created and returned by the ir_to_jit_wrapper rule.",
    fields = {
        "source_file": "File: The source file.",
        "header_file": "File: The header file.",
    },
)

_ir_to_jit_wrapper_attrs = {
    "jit_wrapper_args": attr.string_dict(
        doc = "Arguments of the JIT wrapper tool.",
    ),
    "source_file": attr.output(
        doc = "The generated source file.",
    ),
    "header_file": attr.output(
        doc = "The generated header file.",
    ),
    "_ir_to_jit_wrapper_tool": attr.label(
        doc = "The target of the JIT wrapper executable.",
        default = Label("//xls/jit:jit_wrapper_generator_main"),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
}

def _ir_to_jit_wrapper_impl(ctx):
    """The implementation of the 'ir_to_jit_wrapper' rule.

    Execute the JIT wrapper tool on the IR file.

    Args:
      ctx: The current rule's context object.

    Returns:
      IRToJitWrapperInfo provider
      DefaultInfo provider
    """

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
            jit_wrapper_flags.add("--{}".format(flag_name), jit_wrapper_args[flag_name])
        else:
            fail("Unrecognized argument: %s." % flag_name)

    # source file
    src = ctx.file.src
    jit_wrapper_flags.add("--ir_path", src.path)

    # output filename
    name = ctx.attr.name
    jit_wrapper_flags.add("--output_name", name)
    cc_file = ctx.actions.declare_file(name + ".cc")
    h_file = ctx.actions.declare_file(name + ".h")

    # output directory
    jit_wrapper_flags.add("--output_dir", cc_file.dirname)

    # genfiles directory
    jit_wrapper_flags.add("--genfiles_dir", ctx.genfiles_dir.path)
    my_generated_files = [cc_file, h_file]
    ctx.actions.run(
        outputs = my_generated_files,
        tools = [ctx.executable._ir_to_jit_wrapper_tool],
        inputs = [src, ctx.executable._ir_to_jit_wrapper_tool],
        arguments = [jit_wrapper_flags],
        executable = ctx.executable._ir_to_jit_wrapper_tool.path,
        mnemonic = "IRJITWrapper",
        progress_message = "Building JIT wrapper for source file: %s" % (src.path),
    )
    return [
        IRToJitWrapperInfo(
            source_file = cc_file,
            header_file = h_file,
        ),
        DefaultInfo(
            files = depset(my_generated_files),
        ),
    ]

ir_to_jit_wrapper = rule(
    doc = """A build rule that generates the sources for JIT invocation wrappers.

        Example:

         1) A file as the source.

        ```
            ir_to_jit_wrapper(
                name = "a_jit_wrapper",
                src = "a.ir",
            )
        ```

        2) An ir_opt target as the source.

        ```
            ir_opt(
                name = "a",
                src = "a.x",
            )


            ir_to_jit_wrapper(
                name = "a_jit_wrapper",
                src = ":a",
            )
        ```
    """,
    implementation = _ir_to_jit_wrapper_impl,
    attrs = dict(ir_common_attrs.items() + _ir_to_jit_wrapper_attrs.items()),
)

def cc_ir_to_jit_wrapper(
        name,
        src = None,
        jit_wrapper_args = None):
    """Instantiates ir_to_jit_wrapper and a cc_library target with the files.

    The macro generates sources files (.cc and .h) using the
    ir_to_jit_wrapper rule. The source files are the input to a cc_library
    target with the same name as this macro.

    Args:
      name: The name of the cc_library target.
      src: The path to the IR file.
      jit_wrapper_args: Arguments of the JIT wrapper tool.
    """
    if jit_wrapper_args and not type(jit_wrapper_args) != dict:
        fail("JIT Wrapper arguments must be a dictionary.")
    if src:
        if not type(src) != str:
            fail("The source must be a string.")
    else:
        fail("The source must be defined.")
    filename = "__" + name + "_ir_to_jit_wrapper"
    ir_to_jit_wrapper(
        name = filename,
        src = src,
        jit_wrapper_args = jit_wrapper_args,
        source_file = filename + ".cc",
        header_file = filename + ".h",
    )
    native.cc_library(
        name = name,
        srcs = [":" + filename + ".h", ":" + filename + ".cc"],
        deps = [
            "@com_google_absl//absl/status",
            "//xls/common/status:status_macros",
            "@com_google_absl//absl/status:statusor",
            "//xls/ir",
            "//xls/ir:ir_parser",
            "//xls/ir:value",
            "//xls/ir:value_view",
            "//xls/jit:ir_jit",
        ],
    )
