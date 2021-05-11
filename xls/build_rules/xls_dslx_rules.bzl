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

"""This module contains DSLX-related build rules for XLS."""

load("//xls/build_rules:xls_common_rules.bzl", "get_args")

DEFAULT_DSLX_TEST_ARGS = {
    "compare": "none",
}

DslxFilesInfo = provider(
    doc = "A provider containing DSLX file information for the target. It is " +
          "created and returned by the dslx_library rule.",
    fields = {
        "dslx_sources": "Depset: A depset containing the DSLX source (.x) " +
                        "files of the target and its dependencies.",
        "dslx_dummy_files": "Depset: A depset containing the DSLX generated " +
                            "dummy (.dummy) files of the target and its " +
                            "dependencies. A DSLX dummy file is generated " +
                            "when a DSLX file is successfully parsed and " +
                            "type checked. It is used to create a dependency " +
                            "between dslx_library targets.",
    },
)

def get_transitive_dslx_srcs_files_depset(srcs, deps):
    """Returns a depset representing the transitive DSLX source files.

    The macro is used to collect the transitive DSLX source files of a target.

    Args:
      srcs: a list of DSLX source files (.x)
      deps: a list of targets

    Returns:
      A depset collection where the files from 'srcs' are placed in the 'direct'
      field of the depset and the DSLX source files for each dependency in
      'deps' are placed in the 'transitive' field of the depset.
    """
    return depset(
        srcs,
        transitive = [dep[DslxFilesInfo].dslx_sources for dep in deps],
    )

def get_transitive_dslx_dummy_files_depset(srcs, deps):
    """Returns a depset representing the transitive DSLX dummy files.

    The macro is used to collect the transitive DSLX dummy files of a target.

    Args:
      srcs: a list of DSLX dummy files (.dummy)
      deps: a list of targets dependencies

    Returns:
      A depset collection where the files from 'srcs' are placed in the 'direct'
      field of the depset and the DSLX dummy files for each dependency in 'deps'
      are placed in the 'transitive' field of the depset.
    """
    return depset(
        srcs,
        transitive = [dep[DslxFilesInfo].dslx_dummy_files for dep in deps],
    )

#TODO(https://github.com/google/xls/issues/392) 04-14-21
def parse_and_type_check(ctx, src, required_files):
    """Parses and type checks a file.

    The macro creates an action in the context that parses and type checks a
    file.

    Args:
      ctx: The current rule's context object.
      src: The source file.
      required_files: A list of DSLX sources files required to
        perform the parse and type check action.

    Returns:
      A File referencing the dummy file.
    """
    file = ctx.actions.declare_file(src.basename + ".dummy")
    ctx.actions.run_shell(
        outputs = [file],
        # The DSLX interpreter executable is a tool needed by the action.
        tools = [ctx.executable._dslx_interpreter_tool],
        # The files required for parsing and type checking also requires the
        # DSLX interpreter executable.
        inputs = required_files + [ctx.executable._dslx_interpreter_tool],
        # Generate a dummy file for the DSLX source file when the source file is
        # successfully parsed and type checked.
        command = "\n".join([
            "{} {} --compare=none --execute=false".format(
                ctx.executable._dslx_interpreter_tool.path,
                src.path,
            ),
            "if [ $? -ne 0 ]; then",
            "exit -1",
            "fi",
            "touch {}".format(file.path),
            "exit 0",
        ]),
        mnemonic = "ParseAndTypeCheckDSLXSourceFile",
        progress_message = "Parsing and type checking DSLX source file: %s" %
                           (src.path),
    )
    return file

def get_dslx_test_cmd(ctx, src):
    """Returns the runfiles and command that executes in the dslx_test rule.

    Args:
      ctx: The current rule's context object.
      src: The file to test.

    Returns:
      A tuple with two elements. The files element is a list of runfiles to
      execute the command. The second element is the command.
    """
    dslx_test_default_args = DEFAULT_DSLX_TEST_ARGS
    dslx_test_args = ctx.attr.dslx_test_args
    DSLX_TEST_FLAGS = (
        "compare",
    )

    my_args = get_args(dslx_test_args, DSLX_TEST_FLAGS, dslx_test_args)
    cmd = "{} {} {}".format(
        ctx.executable._dslx_interpreter_tool.short_path,
        src.path,
        my_args,
    )

    # The required runfiles are the source file, the DSLX interpreter executable
    # and the DSLX std library.
    runfiles = [src, ctx.executable._dslx_interpreter_tool]
    runfiles += ctx.files._dslx_std_lib
    return runfiles, cmd

# Common attributes for the dslx_library and dslx_test rules.
_dslx_common_attrs = {
    "deps": attr.label_list(
        doc = "Dependency targets for the rule.",
        providers = [DslxFilesInfo],
    ),
}
dslx_exec_attrs = {
    "_dslx_std_lib": attr.label(
        doc = "The target containing the DSLX std library.",
        default = Label("//xls/dslx/stdlib:dslx_std_lib"),
        cfg = "target",
    ),
    "_dslx_interpreter_tool": attr.label(
        doc = "The target of the DSLX interpreter executable.",
        default = Label("//xls/dslx:interpreter_main"),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
}

# Attributes for the dslx_library rule.
_dslx_library_attrs = {
    "srcs": attr.label_list(
        doc = "Source files for the rule. Files must have a '.x' extension.",
        allow_files = [".x"],
    ),
}

# Attributes for the dslx_test rule.
_dslx_test_attrs = {
    "src": attr.label(
        doc = "The DSLX source file for the rule. A single source file must " +
              "be provided. The file must have a '.x' extension.",
        mandatory = True,
        allow_single_file = [".x"],
    ),
}

# Common attributes for DSLX testing.
dslx_test_common_attrs = {
    "dslx_test_args": attr.string_dict(
        doc = "Arguments of the DSLX interpreter executable.",
    ),
}

def _dslx_library_impl(ctx):
    """The implementation of the 'dslx_library' rule.

    Parses and type checks DSLX source files. When a DSLX file is successfully
    parsed and type checked, a DSLX dummy file is generated. The dummy file is
    used to create a dependency between the current target and targets
    descendents.

    Args:
      ctx: The current rule's context object.

    Returns:
      DslxFilesInfo provider
      DefaultInfo provider
    """
    my_srcs_list = ctx.files.srcs
    my_dummy_files = []
    my_srcs_depset = get_transitive_dslx_srcs_files_depset(
        ctx.files.srcs + ctx.files._dslx_std_lib,
        ctx.attr.deps,
    )

    # The required files are the source files from the current target and its
    # transitive dependencies
    required_files = my_srcs_depset.to_list()

    # Parse and type check each source file
    for src in my_srcs_list:
        file = parse_and_type_check(ctx, src, required_files)
        my_dummy_files.append(file)

    dummy_files_depset = get_transitive_dslx_dummy_files_depset(
        my_dummy_files,
        ctx.attr.deps,
    )
    return [
        DslxFilesInfo(
            dslx_sources = my_srcs_depset,
            dslx_dummy_files = dummy_files_depset,
        ),
        DefaultInfo(files = dummy_files_depset),
    ]

dslx_library = rule(
    doc = """
        A build rule that parses and type checks DSLX source files.

        Examples:

        1) A collection of DSLX source files.

        ```
        dslx_library(
            name = "files_123",
            srcs = [
                "file_1.x",
                "file_2.x",
                "file_3.x",
            ],
        )
        ```

        2) Dependency on other dslx_library targets.

        ```
        dslx_library(
            name = "a",
            srcs = [
                "a.x",
            ],
        )

        # Depends on target a.
        dslx_library(
            name = "b",
            srcs = [
                "b.x",
            ],
            deps = [
                ":a",
            ],
        )

        # Depends on target a.
        dslx_library(
            name = "c",
            srcs = [
                "c.x",
            ],
            deps = [
                ":a",
            ],
        )
        ```
    """,
    implementation = _dslx_library_impl,
    attrs = dict(
        _dslx_common_attrs.items() +
        _dslx_library_attrs.items() +
        dslx_exec_attrs.items(),
    ),
)

def _dslx_test_impl(ctx):
    """The implementation of the 'dslx_test' rule.

    Executes the tests and quick checks of a DSLX source file.

    Args:
      ctx: The current rule's context object.

    Returns:
      DefaultInfo provider
    """
    src = ctx.file.src
    runfiles, cmd = get_dslx_test_cmd(ctx, src)

    # The runfiles also require the source files from its transitive
    # dependencies.
    for dep in ctx.attr.deps:
        runfiles += dep[DslxFilesInfo].dslx_sources.to_list()

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

dslx_test = rule(
    doc = """
        A dslx test executes the tests and quick checks of a DSLX source file.

        Example:
        ```
            # Assume a dslx_library target c is present.
            dslx_test(
                name = "d_test",
                src = "d.x",
                deps = [
                    ":c",
                ],
            )
        ```
    """,
    implementation = _dslx_test_impl,
    attrs = dict(
        _dslx_common_attrs.items() +
        _dslx_test_attrs.items() +
        dslx_test_common_attrs.items() +
        dslx_exec_attrs.items(),
    ),
    test = True,
)
