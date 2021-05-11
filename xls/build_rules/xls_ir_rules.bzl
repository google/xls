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

load("//xls/build_rules:xls_common_rules.bzl", "get_args")
load("//xls/build_rules:xls_dslx_rules.bzl", "DslxFilesInfo")

DEFAULT_IR_EVAL_TEST_ARGS = {
    "random_inputs": "100",
    "optimize_ir": "true",
}

def convert_to_ir(ctx, src):
    """Converts a DSLX source file to an IR file.

    Creates an action in the context that converts a DSLX source file to an
    IR file.

    Args:
      ctx: The current rule's context object.
      src: The source file.
    Returns:
      A File referencing the IR file.
    """
    ir_conv_args = ctx.attr.ir_conv_args
    IR_CONV_FLAGS = (
        "entry",
        "dslx_path",
    )

    my_args = get_args(ir_conv_args, IR_CONV_FLAGS)

    required_files = ctx.files._dslx_std_lib + [src]
    for dep in ctx.attr.deps:
        required_files += dep[DslxFilesInfo].dslx_sources.to_list()

    ir_file = ctx.actions.declare_file(src.basename[:-1] + "ir")
    ctx.actions.run_shell(
        outputs = [ir_file],
        # The IR converter executable is a tool needed by the action.
        tools = [ctx.executable._ir_converter_tool],
        # The files required for converting the DSLX source file.
        inputs = required_files + [ctx.executable._ir_converter_tool],
        command = "{} {} {} > {}".format(
            ctx.executable._ir_converter_tool.path,
            my_args,
            src.path,
            ir_file.path,
        ),
        mnemonic = "ConvertDSLX",
        progress_message = "Converting DSLX file: %s" % (src.path),
    )
    return ir_file

def optimize_ir(ctx, src):
    """Optimizes an IR file.

    Creates an action in the context that optimizes an IR file.

    Args:
      ctx: The current rule's context object.
      src: The source file.

    Returns:
      A File referencing the optimized IR file.
    """
    ir_opt_args = ctx.attr.ir_opt_args
    IR_OPT_FLAGS = (
        "entry",
        "ir_dump_path",
        "run_only_passes",
        "skip_passes",
        "opt_level",
    )

    my_args = get_args(ir_opt_args, IR_OPT_FLAGS)

    opt_ir_file = ctx.actions.declare_file(src.basename[:-2] + "opt.ir")
    ctx.actions.run_shell(
        outputs = [opt_ir_file],
        # The IR optimization executable is a tool needed by the action.
        tools = [ctx.executable._ir_opt_tool],
        # The files required for optimizing the IR file.
        inputs = [src, ctx.executable._ir_opt_tool],
        command = "{} {} {} > {}".format(
            ctx.executable._ir_opt_tool.path,
            src.path,
            my_args,
            opt_ir_file.path,
        ),
        mnemonic = "OptimizeIR",
        progress_message = "Optimizing IR file: %s" % (src.path),
    )
    return opt_ir_file

def get_ir_equivalence_test_cmd(ctx, src_0, src_1):
    """Returns the runfiles and command that executes in the ir_equivalence_test rule.

    Args:
      ctx: The current rule's context object.
      src_0: A file for the test.
      src_1: A file for the test.

    Returns:
      A tuple with two elements. The files element is a list of runfiles to
      execute the command. The second element is the command.
    """
    ir_equivalence_args = ctx.attr.ir_equivalence_args
    IR_EQUIVALENCE_FLAGS = (
        "function",
        "timeout",
    )

    my_args = get_args(ir_equivalence_args, IR_EQUIVALENCE_FLAGS)

    cmd = "{} {} {} {}\n".format(
        ctx.executable._ir_equivalence_tool.short_path,
        src_0.short_path,
        src_1.short_path,
        my_args,
    )

    # The required runfiles are the source files and the IR equivalence tool
    # executable.
    runfiles = [src_0, src_1, ctx.executable._ir_equivalence_tool]
    return runfiles, cmd

def get_ir_eval_test_cmd(ctx, src):
    """Returns the runfiles and command that executes in the ir_eval_test rule.

    Args:
      ctx: The current rule's context object.
      src: The file to test.

    Returns:
      A tuple with two elements. The files element is a list of runfiles to
      execute the command. The second element is the command.
    """
    ir_eval_default_args = DEFAULT_IR_EVAL_TEST_ARGS
    ir_eval_args = ctx.attr.ir_eval_args
    IR_EVAL_FLAGS = (
        "entry",
        "input",
        "input_file",
        "random_inputs",
        "expected",
        "expected_file",
        "optimize_ir",
        "eval_after_each_pass",
        "use_llvm_jit",
        "test_llvm_jit",
        "llvm_opt_level",
        "test_only_inject_jit_result",
    )

    my_args = get_args(ir_eval_args, IR_EVAL_FLAGS, ir_eval_default_args)

    cmd = "{} {} {}".format(
        ctx.executable._ir_eval_tool.short_path,
        src.short_path,
        my_args,
    )

    # The required runfiles are the source file and the IR interpreter tool
    # executable.
    runfiles = [src, ctx.executable._ir_eval_tool]
    return runfiles, cmd

def get_ir_benchmark_cmd(ctx, src):
    """Returns the runfiles and command that executes in the ir_benchmark rule.

    Args:
      ctx: The current rule's context object.
      src: The file to benchmark.

    Returns:
      A tuple with two elements. The files element is a list of runfiles to
      execute the command. The second element is the command.
    """
    benchmark_args = ctx.attr.benchmark_args
    BENCHMARK_FLAGS = (
        "clock_period_ps",
        "pipeline_stages",
        "clock_margin_percent",
        "show_known_bits",
        "entry",
        "delay_model",
    )

    my_args = get_args(benchmark_args, BENCHMARK_FLAGS)

    cmd = "{} {} {}".format(
        ctx.executable._benchmark_tool.short_path,
        src.short_path,
        my_args,
    )

    # The required runfiles are the source files and the IR benchmark tool
    # executable.
    runfiles = [src, ctx.executable._benchmark_tool]
    return runfiles, cmd

def _dslx_to_ir_impl(ctx):
    """The implementation of the 'dslx_to_ir' rule.

    Converts a DSLX source file to an IR file.

    Args:
    ctx: The current rule's context object.

    Returns:
    DefaultInfo provider
    """
    src = ctx.file.src
    ir_file = convert_to_ir(ctx, src)
    return [
        DefaultInfo(files = depset([ir_file])),
    ]

ir_common_attrs = {
    "src": attr.label(
        doc = "The IR source file for the rule. A single source file must be " +
              "provided. The file must have a '.ir' extension.",
        mandatory = True,
        allow_single_file = [".ir"],
    ),
}

dslx_to_ir_attrs = {
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
    "ir_conv_args": attr.string_dict(
        doc = "Arguments of the IR conversion tool.",
    ),
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
                ir_conv_args = {
                    "entry" : "a",
                },
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
    attrs = dslx_to_ir_attrs,
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
    opt_ir_file = optimize_ir(ctx, ir_file)

    return [
        DefaultInfo(files = depset([opt_ir_file])),
    ]

ir_opt_attrs = {
    "ir_opt_args": attr.string_dict(
        doc = "Arguments of the IR optimizer tool.",
    ),
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
                ir_opt_args = {
                    "entry" : "a",
                },
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
    attrs = dict(ir_common_attrs.items() + ir_opt_attrs.items()),
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
    runfiles, cmd = get_ir_equivalence_test_cmd(ctx, ir_file_a, ir_file_b)
    executable_file = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(
        output = executable_file,
        content = "\n".join([
            "#!/bin/bash",
            "set -e",
            cmd,
            "exit 0",
        ]),
        is_executable = True,
    )

    return [
        DefaultInfo(
            runfiles = ctx.runfiles(files = runfiles),
            files = depset([executable_file]),
            executable = executable_file,
        ),
    ]

_two_ir_files_attrs = {
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
}

ir_equivalence_test_attrs = {
    "ir_equivalence_args": attr.string_dict(
        doc = "Arguments of the IR equivalence tool.",
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
    attrs = dict(_two_ir_files_attrs.items() + ir_equivalence_test_attrs.items()),
    test = True,
)

def _ir_eval_test_impl(ctx):
    """The implementation of the 'ir_eval_test' rule.

    Executes the IR Interpreter on an IR file.

    Args:
      ctx: The current rule's context object.
    Returns:
      DefaultInfo provider
    """
    src = ctx.file.src
    runfiles, cmd = get_ir_eval_test_cmd(ctx, src)
    executable_file = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(
        output = executable_file,
        content = "\n".join([
            "#!/bin/bash",
            "set -e",
            cmd,
            "exit 0",
        ]),
        is_executable = True,
    )

    return [
        DefaultInfo(
            runfiles = ctx.runfiles(files = runfiles),
            files = depset([executable_file]),
            executable = executable_file,
        ),
    ]

ir_eval_test_attrs = {
    "ir_eval_args": attr.string_dict(
        doc = "Arguments of the IR interpreter.",
        default = DEFAULT_IR_EVAL_TEST_ARGS,
    ),
    "_ir_eval_tool": attr.label(
        doc = "The target of the IR interpreter executable.",
        default = Label("//xls/tools:eval_ir_main"),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
}

ir_eval_test = rule(
    doc = """A IR evaluation test executes the IR interpreter on an IR file.

        Example:

         1) A file as the source.

        ```
            ir_eval_test(
                name = "a_test",
                src = "a.ir",
            )
        ```

        2) An ir_opt target as the source.

        ```
            ir_opt(
                name = "a",
                src = "a.x",
            )


            ir_eval_test(
                name = "a_test",
                src = ":a",
            )
        ```
    """,
    implementation = _ir_eval_test_impl,
    attrs = dict(ir_common_attrs.items() + ir_eval_test_attrs.items()),
    test = True,
)

def _ir_benchmark_impl(ctx):
    """The implementation of the 'ir_benchmark' rule.

    Executes the benchmark tool on an IR file.

    Args:
      ctx: The current rule's context object.
    Returns:
      DefaultInfo provider
    """
    src = ctx.file.src
    runfiles, cmd = get_ir_benchmark_cmd(ctx, src)
    executable_file = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(
        output = executable_file,
        content = "\n".join([
            "#!/bin/bash",
            "set -e",
            cmd,
            "exit 0",
        ]),
        is_executable = True,
    )

    return [
        DefaultInfo(
            runfiles = ctx.runfiles(files = runfiles),
            files = depset([executable_file]),
            executable = executable_file,
        ),
    ]

ir_benchmark_attrs = {
    "benchmark_args": attr.string_dict(
        doc = "Arguments of the benchmark tool.",
    ),
    "_benchmark_tool": attr.label(
        doc = "The target of the benchmark executable.",
        default = Label("//xls/tools:benchmark_main"),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
}

ir_benchmark = rule(
    doc = """A IR benchmark executes the benchmark tool on an IR file.

        Example:

         1) A file as the source.

        ```
            ir_benchmark(
                name = "a_benchmark",
                src = "a.ir",
            )
        ```

        2) An ir_opt target as the source.

        ```
            ir_opt(
                name = "a",
                src = "a.x",
            )


            ir_benchmark(
                name = "a_benchmark",
                src = ":a",
            )
        ```
    """,
    implementation = _ir_benchmark_impl,
    attrs = dict(ir_common_attrs.items() + ir_benchmark_attrs.items()),
    executable = True,
)
