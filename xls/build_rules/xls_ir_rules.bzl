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
"""This module contains IR-related build rules for XLS."""

load("//xls/build_rules:xls_dslx_rules.bzl", "DslxFilesInfo")

def _convert_to_ir(ctx, src, required_files):
    """Converts a DSLX source file to an IR file.

    The macro creates an action in the context that converts a DSLX source
    file
    to an IR file.

    Args:
      ctx: The current rule's context object.
      src: The source file.
      required_files: A list of DSLX sources files required to perform the
        conversion action.

    Returns:
      A File referencing the IR file.
    """
    entry = ctx.attr.entry
    args = ("--entry=" + entry) if entry else ""
    ir_file = ctx.actions.declare_file(src.basename[:-1] + "ir")
    ctx.actions.run_shell(
        outputs = [ir_file],
        # The IR converter executable is a tool needed by the action.
        tools = [ctx.executable._ir_converter_tool],
        # The files required for converting the DSLX source file also requires
        # the IR converter executable.
        inputs = required_files + [ctx.executable._ir_converter_tool],
        command = "{} {} {} > {}".format(
            ctx.executable._ir_converter_tool.path,
            args,
            src.path,
            ir_file.path,
        ),
        mnemonic = "ConvertDSLX",
        progress_message = "Converting DSLX file: %s" % (src.path),
    )
    return ir_file

def _optimize_ir(ctx, src):
    """Optimizes an IR file.

    The macro creates an action in the context that optimizes an IR file.

    Args:
      ctx: The current rule's context object.
      src: The source file.

    Returns:
      A File referencing the optimized IR file.
    """
    entry = ctx.attr.entry
    args = ("--entry=" + entry) if entry else ""
    opt_ir_file = ctx.actions.declare_file(src.basename[:-2] + "opt.ir")
    ctx.actions.run_shell(
        outputs = [opt_ir_file],
        # The IR optimization executable is a tool needed by the action.
        tools = [ctx.executable._ir_opt_tool],
        # The files required for optimizing the IR file also requires the IR
        # optimization executable.
        inputs = [src, ctx.executable._ir_opt_tool],
        command = "{} {} {} > {}".format(
            ctx.executable._ir_opt_tool.path,
            src.path,
            args,
            opt_ir_file.path,
        ),
        mnemonic = "OptimizeIR",
        progress_message = "Optimizing IR file: %s" % (src.path),
    )
    return opt_ir_file

def _dslx_to_ir_impl(ctx):
    """The implementation of the 'dslx_to_ir' rule.

    Converts a DSLX source file to an IR file.

    Args:
    ctx: The current rule's context object.

    Returns:
    DefaultInfo provider
    """
    src = ctx.file.src
    my_srcs = [
        item
        for dep in ctx.attr.deps
        for item in dep[DslxFilesInfo].dslx_sources.to_list()
    ]
    my_srcs += [src] + ctx.files._dslx_std_lib
    ir_file = _convert_to_ir(ctx, src, my_srcs)
    return [
        DefaultInfo(files = depset([ir_file])),
    ]

_dslx_to_ir_attrs = {
    "src": attr.label(
        doc = "The DSLX source file for the rule. A single source file must " +
              "be provided. The file must have a '.x' extension.",
        mandatory = True,
        allow_single_file = [".x"],
    ),
    "deps": attr.label_list(
        doc = "Dependency targets for the rule.",
        providers = [DslxFilesInfo],
    ),
    "entry": attr.string(doc = "Entry function name for conversion."),
    "_dslx_std_lib": attr.label(
        doc = "The target containing the DSLX std library.",
        default = Label("//xls/dslx/stdlib:dslx_std_lib"),
        cfg = "target",
    ),
    "_ir_converter_tool": attr.label(
        doc = "The target of the IR converter executable.",
        default = Label("//xls/dslx:ir_converter_main"),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
}

dslx_to_ir = rule(
    doc = """
        A build rule that converts a DSLX source file to an IR file.

        Examples:

        1) An IR conversion with an entry defined.

        ```
            dslx_to_ir(
                name = "a_dslx_to_ir",
                src = "a.x",
                entry = "a",
            )
        ```

        2) An IR conversion with dependency on dslx_library targets.

        ```
            dslx_library(
                name = "files_ab",
                srcs = [
                    "a.x",
                    "b.x",
                ],
            )

            dslx_library(
                name = "c",
                srcs = [
                    "c.x",
                ],
            )

            dslx_to_ir(
                name = "d_dslx_to_ir",
                src = "d.x",
                deps = [
                    ":files_ab",
                    ":c",
                ],
            )
        ```
    """,
    implementation = _dslx_to_ir_impl,
    attrs = _dslx_to_ir_attrs,
)

def _ir_opt_impl(ctx):
    """The implementation of the 'ir_opt' rule.

    Optimizes an IR file.

    Args:
    ctx: The current rule's context object.

    Returns:
    DefaultInfo provider
    """
    ir_file = ctx.file.src
    opt_ir_file = _optimize_ir(ctx, ir_file)

    return [
        DefaultInfo(files = depset([opt_ir_file])),
    ]

_ir_opt_attrs = {
    "src": attr.label(
        doc = "The IR source file for the rule. A single source file must be " +
              "provided. The file must have a '.ir' extension.",
        mandatory = True,
        allow_single_file = [".ir"],
    ),
    "entry": attr.string(doc = "Entry function name for the optimization."),
    "_ir_opt_tool": attr.label(
        doc = "The target of the IR optimizer executable.",
        default = Label("//xls/tools:opt_main"),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
}

ir_opt = rule(
    doc = """
        A build rule that optimizes an IR file.

        Examples:

        1) Optimizing an IR file with an entry defined.

        ```
            ir_opt(
                name = "a_ir_opt",
                src = "a.ir",
                entry = "a",
            )
        ```

        2) A target as the source.

        ```
            dslx_to_ir(
                name = "a",
                src = "a.x",
            )
            ir_opt(
                name = "a_ir_opt",
                src = ":a",
            )
        ```
    """,
    implementation = _ir_opt_impl,
    attrs = _ir_opt_attrs,
)

def _ir_equivalence_test_impl(ctx):
    """The implementation of the 'ir_equivalence_test' rule.

    Executes the equivalence tool on two IR files.

    Args:
    ctx: The current rule's context object.

    Returns:
    DefaultInfo provider
    """
    ir_file_a = ctx.file.src_0
    ir_file_b = ctx.file.src_1
    entry = ctx.attr.entry
    args = ("--function=" + entry) if entry else ""

    executable_file = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(
        output = executable_file,
        content = "\n".join([
            "#!/bin/bash",
            "set -e",
            "{} {} {} {}\n".format(
                ctx.executable._ir_equivalence_tool.short_path,
                ir_file_a.short_path,
                ir_file_b.short_path,
                args,
            ),
            "exit 0",
        ]),
        is_executable = True,
    )

    runfiles = [ir_file_a, ir_file_b, ctx.executable._ir_equivalence_tool]

    return [
        DefaultInfo(
            runfiles = ctx.runfiles(files = runfiles),
            files = depset([executable_file]),
            executable = executable_file,
        ),
    ]

_ir_equivalence_test_attrs = {
    "src_0": attr.label(
        doc = "An IR source file for the rule. A single source file must be " +
              "provided. The file must have a '.ir' extension.",
        mandatory = True,
        allow_single_file = [".ir"],
    ),
    "src_1": attr.label(
        doc = "An IR source file for the rule. A single source file must be " +
              "provided. The file must have a '.ir' extension.",
        mandatory = True,
        allow_single_file = [".ir"],
    ),
    "entry": attr.string(
        doc = "Entry function name for the optimization test.",
    ),
    "_ir_equivalence_tool": attr.label(
        doc = "The target of the IR equivalence executable.",
        default = Label("//xls/tools:check_ir_equivalence_main"),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
}

ir_equivalence_test = rule(
    doc = """
        An IR equivalence test executes executes the equivalence tool on two IR
        files.

        Example:

         1) A file as the source.

        ```
            ir_equivalence_test(
                name = "ab_test",
                src_0 = "a.ir",
                src_1 = "b.ir",
            )
        ```

        2) A target as the source.
        ```
            dslx_to_ir (
                name = "b",
                src = "b.x",
            )

            ir_equivalence_test(
                name = "ab_test",
                src_0 = "a.ir",
                src_1 = ":b",
            )
        ```

    """,
    implementation = _ir_equivalence_test_impl,
    attrs = _ir_equivalence_test_attrs,
    test = True,
)
