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

load("@bazel_skylib//lib:dicts.bzl", "dicts")
load(
    "//xls/build_rules:xls_common_rules.bzl",
    "append_cmd_line_args_to",
    "get_args",
)
load(
    "//xls/build_rules:xls_providers.bzl",
    "DslxInfo",
)
load(
    "//xls/build_rules:xls_toolchains.bzl",
    "get_xls_toolchain_info",
    "xls_toolchain_attr",
)

_DEFAULT_DSLX_TEST_ARGS = {
    "compare": "jit",
}

def get_transitive_dslx_srcs_files_depset(srcs, deps, stdlib):
    """Returns a depset representing the transitive DSLX source files.

    The macro is used to collect the transitive DSLX source files of a target.

    Args:
      srcs: a list of DSLX source files (.x)
      deps: a list of targets
      stdlib: Target (as in Bazel Target object) containing the DSLX stdlib
          files.

    Returns:
      A depset collection where the files from 'srcs' are placed in the 'direct'
      field of the depset and the DSLX source files for each dependency in
      'deps' are placed in the 'transitive' field of the depset.
    """
    return depset(
        srcs,
        transitive = [dep[DslxInfo].dslx_source_files for dep in deps] +
                     [stdlib.files],
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
        transitive = [dep[DslxInfo].dslx_dummy_files for dep in deps],
    )

#TODO(https://github.com/google/xls/issues/392) 04-14-21
def parse_and_type_check(ctx, srcs, required_files):
    """Parses and type checks a list containing DSLX files.

    The macro creates an action in the context that parses and type checks a
    list containing DSLX files.

    Args:
      ctx: The current rule's context object.
      srcs: A list of DSLX files.
      required_files: A list of DSLX sources files required to
        perform the parse and type check action.

    Returns:
      A File referencing the dummy file.
    """
    dslx_interpreter_tool = get_xls_toolchain_info(ctx).dslx_interpreter_tool
    dslx_srcs_str = " ".join([s.path for s in srcs])
    file = ctx.actions.declare_file(ctx.attr.name + ".dummy")
    ctx.actions.run_shell(
        outputs = [file],
        # The DSLX interpreter executable is a tool needed by the action.
        tools = [dslx_interpreter_tool],
        # The files required for parsing and type checking also requires the
        # DSLX interpreter executable.
        inputs = required_files + [dslx_interpreter_tool],
        # Generate a dummy file for the DSLX source file when the source file is
        # successfully parsed and type checked.
        # TODO (vmirian) 01-05-21 Enable the interpreter to take multiple files.
        # TODO (vmirian) 01-05-21 Ideally, create a standalone tool that parses
        # a DSLX file. (Instead of repurposing the interpreter.)
        command = "\n".join([
            "FILES=\"{}\"".format(dslx_srcs_str),
            "for file in $FILES; do",
            "{} $file --compare=none --execute=false --dslx_path={}".format(
                dslx_interpreter_tool.path,
                ":${PWD}:" + ctx.genfiles_dir.path + ":" + ctx.bin_dir.path,
            ),
            "if [ $? -ne 0 ]; then",
            "echo \"Error parsing and type checking DSLX source file: $file\"",
            "exit -1",
            "fi",
            "done",
            "touch {}".format(file.path),
            "exit 0",
        ]),
        mnemonic = "ParseAndTypeCheckDSLXSourceFile",
        progress_message = "Parsing and type checking DSLX source files of " +
                           "target %s" % (ctx.attr.name),
    )
    return file

def _get_dslx_test_cmdline(ctx, src, append_cmd_line_args = True):
    """Returns the command that executes in the xls_dslx_test rule.

    Args:
      ctx: The current rule's context object.
      src: The file to test.
      append_cmd_line_args: Flag controlling appending the command-line
        arguments invoking the command generated by this function. When set to
        True, the command-line arguments invoking the command are appended.
        Otherwise, the command-line arguments are not appended.

    Returns:
      The command that executes in the xls_dslx_test rule.
    """
    dslx_interpreter_tool = get_xls_toolchain_info(ctx).dslx_interpreter_tool
    dslx_test_default_args = _DEFAULT_DSLX_TEST_ARGS
    _dslx_test_args = ctx.attr.dslx_test_args
    DSLX_TEST_FLAGS = (
        "bytecode",
        "compare",
        "dslx_path",
    )

    dslx_test_args = dict(_dslx_test_args)
    dslx_test_args["dslx_path"] = (
        dslx_test_args.get("dslx_path", "") + ":${PWD}:" +
        ctx.genfiles_dir.path + ":" + ctx.bin_dir.path
    )
    my_args = get_args(dslx_test_args, DSLX_TEST_FLAGS, dslx_test_default_args)

    cmd = "{} {} {}".format(
        dslx_interpreter_tool.short_path,
        src.short_path,
        my_args,
    )

    # Append command-line arguments.
    if append_cmd_line_args:
        cmd = append_cmd_line_args_to(cmd)

    return cmd

def get_files_from_dslx_library_as_input(ctx):
    """Returns the DSLX source files and transitive files of rules using 'xls_dslx_library_as_input_attrs'.

    Args:
      ctx: The current rule's context object.

    Returns:
      A tuple with the first element representing the DSLX files and the
      second element representing the transitive DSLX files of the rule.
    """
    dslx_src_files = []
    transitive_files = []
    count = 0

    if ctx.attr.library:
        dslx_info = ctx.attr.library[DslxInfo]
        dslx_src_files = dslx_info.target_dslx_source_files
        transitive_files += dslx_info.dslx_source_files.to_list()
        count += 1
    if ctx.attr.srcs or ctx.attr.deps:
        if not ctx.attr.srcs:
            fail("'srcs' must be defined when 'deps' is defined.")
        dslx_src_files = ctx.files.srcs
        for dep in ctx.attr.deps:
            transitive_files += dep[DslxInfo].dslx_source_files.to_list()
        count += 1

    if count != 1:
        fail("One of: 'library' or ['srcs', 'deps'] must be assigned.")

    return dslx_src_files, transitive_files

# Attributes for the xls_dslx_library rule.
_xls_dslx_library_attrs = {
    "srcs": attr.label_list(
        doc = "Source files for the rule. Files must have a '.x' extension.",
        allow_files = [".x"],
    ),
    "deps": attr.label_list(
        doc = "Dependency targets for the rule.",
        providers = [DslxInfo],
    ),
}

xls_dslx_library_as_input_attrs = {
    "library": attr.label(
        doc = "A DSLX library target where the direct (non-transitive) " +
              "files of the target are tested. This attribute is mutually " +
              "exclusive with the 'srcs' and 'deps' attribute.",
        providers = [DslxInfo],
    ),
    "srcs": attr.label_list(
        doc = "Source files for the rule. The files must have a '.x' " +
              "extension. This attribute is mutually exclusive with the " +
              "'library' attribute.",
        allow_files = [".x"],
    ),
    "deps": attr.label_list(
        doc = "Dependency targets for the files in the 'srcs' attribute. " +
              "This attribute is mutually exclusive with the 'library' " +
              "attribute.",
        providers = [DslxInfo],
    ),
}

# Common attributes for DSLX testing.
xls_dslx_test_common_attrs = {
    "dslx_test_args": attr.string_dict(
        doc = "Arguments of the DSLX interpreter executable. For details " +
              "on the arguments, refer to the interpreter_main " +
              "application at " +
              "//xls/dslx/interpreter_main.cc. When the " +
              "default XLS toolchain differs from the default toolchain, " +
              "the application target may be different.",
    ),
}

def _xls_dslx_library_impl(ctx):
    """The implementation of the 'xls_dslx_library' rule.

    Parses and type checks DSLX source files. When a DSLX file is successfully
    parsed and type checked, a DSLX dummy file is generated. The dummy file is
    used to create a dependency between the current target and the target's
    descendants.

    Args:
      ctx: The current rule's context object.

    Returns:
      DslxInfo provider
      DefaultInfo provider
    """
    my_srcs_list = ctx.files.srcs
    my_dummy_files = []
    my_srcs_depset = get_transitive_dslx_srcs_files_depset(
        ctx.files.srcs,
        ctx.attr.deps,
        get_xls_toolchain_info(ctx).dslx_std_lib_target,
    )

    # The required files are the source files from the current target, the
    # standard library files, and its transitive dependencies.
    required_files = my_srcs_depset.to_list()
    required_files += get_xls_toolchain_info(ctx).dslx_std_lib_list

    # Parse and type check the DSLX source files.
    file = parse_and_type_check(ctx, my_srcs_list, required_files)
    my_dummy_files.append(file)

    dummy_files_depset = get_transitive_dslx_dummy_files_depset(
        my_dummy_files,
        ctx.attr.deps,
    )

    runfiles = ctx.runfiles(
        files = ctx.files.srcs,
        transitive_files = get_xls_toolchain_info(ctx).dslx_std_lib_target.files,
    )
    for dep in ctx.attr.deps:
        runfiles = runfiles.merge(dep.default_runfiles)
    return [
        DslxInfo(
            target_dslx_source_files = my_srcs_list,
            dslx_source_files = my_srcs_depset,
            dslx_dummy_files = dummy_files_depset,
        ),
        DefaultInfo(
            files = dummy_files_depset,
            runfiles = runfiles,
        ),
    ]

xls_dslx_library = rule(
    doc = """A build rule that parses and type checks DSLX source files.

        Examples:

        1) A collection of DSLX source files.
            xls_dslx_library(
                name = "files_123_dslx",
                srcs = [
                    "file_1.x",
                    "file_2.x",
                    "file_3.x",
                ],
            )

        2) Dependency on other xls_dslx_library targets.
            xls_dslx_library(
                name = "a_dslx",
                srcs = [
                    "a.x",
                ],
            )

            # Depends on target a_dslx.
            xls_dslx_library(
                name = "b_dslx",
                srcs = [
                    "b.x",
                ],
                deps = [
                    ":a_dslx",
                ],
            )

            # Depends on target a_dslx.
            xls_dslx_library(
                name = "c_dslx",
                srcs = [
                    "c.x",
                ],
                deps = [
                    ":a_dslx",
                ],
            )
    """,
    implementation = _xls_dslx_library_impl,
    attrs = dicts.add(
        _xls_dslx_library_attrs,
        xls_toolchain_attr,
    ),
)

def get_dslx_test_cmd(ctx, src_files_to_test):
    """Returns the runfiles and commands to execute the sources files.

    Args:
      ctx: The current rule's context object.
      src_files_to_test: A list of source files to test..

    Returns:
      A tuple with two elements. The first element is a list of runfiles to
      execute the commands. The second element is a list of commands.

    """

    # The required runfiles are the source files, the DSLX std library and
    # the DSLX interpreter executable.
    runfiles = list(src_files_to_test)
    runfiles += get_xls_toolchain_info(ctx).dslx_std_lib_list
    runfiles.append(get_xls_toolchain_info(ctx).dslx_interpreter_tool)

    cmds = []
    for src in src_files_to_test:
        cmds.append(_get_dslx_test_cmdline(ctx, src))
    return runfiles, cmds

def _xls_dslx_test_impl(ctx):
    """The implementation of the 'xls_dslx_test' rule.

    Executes the tests and quick checks of a DSLX source file.

    Args:
      ctx: The current rule's context object.

    Returns:
      DefaultInfo provider
    """
    src_files_to_test, runfiles = get_files_from_dslx_library_as_input(ctx)

    my_runfiles, cmds = get_dslx_test_cmd(ctx, src_files_to_test)
    runfiles += my_runfiles

    executable_file = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(
        output = executable_file,
        content = "\n".join([
            "#!/bin/bash",
            "set -e",
            "\n".join(cmds),
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

xls_dslx_test = rule(
    doc = """A dslx test executes the tests and quick checks of a DSLX source file.

        Example:

        1) xls_dslx_test on DSLX source files.
            # Assume a xls_dslx_library target bc_dslx is present.
            xls_dslx_test(
                name = "e_dslx_test",
                srcs = [
                    "d.x",
                    "e.x",
                ],
                deps = [":bc_dslx"],
            )

        2) xls_dslx_test on a xls_dslx_library.

            xls_dslx_library(
                name = "b_dslx",
                src = "b.x",
                deps = [
                    ":a_dslx",
                ],
            )

            xls_dslx_test(
                name = "b_dslx_test",
                library = "b_dslx",
            )
    """,
    implementation = _xls_dslx_test_impl,
    attrs = dicts.add(
        xls_dslx_library_as_input_attrs,
        xls_dslx_test_common_attrs,
        xls_toolchain_attr,
    ),
    test = True,
)
