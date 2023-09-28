# Copyright 2023 The XLS Authors
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

"""Generates HTML dashboard that aggregates test results.

User has to provide the output directory for the generated website using
the `-o` argument, for example:

`bazel run -- //xls/examples:dashboard -o my_dashboard_directory`
"""

def _prepare_test_related_attrs(tests):
    """Reads the test attributes passed to the dashboard macro, and converts
    it to the attributes accepted by the underlying bazel rule. Tests and parsers
    are provided as a separate list of labels. All the relations between them and
    additional information about the output files to parse is passed as a format
    string that should be resolved in the underlying bazel rule. Note that
    the provided format string depends on the order of test and parser labels.
    """

    attr_tests = []
    attr_parsers = []
    attr_cmd_args = ""
    for i, test in enumerate(tests):
        test_key = "t{}".format(i)
        attr_tests += [test["name"]]

        output_parsers = test["output_parsers"] if "output_parsers" in test else []
        for parser in output_parsers:
            parseattr_key = "p{}".format(len(attr_parsers))
            attr_parsers += [parser]
            attr_cmd_args += " -p {{{}}},{{{}}}".format(test_key, parseattr_key)

        file_parsers = test["file_parsers"] if "file_parsers" in test else {}
        for parser, file in file_parsers.items():
            parser_key = "p{}".format(len(attr_parsers))
            attr_parsers += [parser]
            attr_cmd_args += " -f {{{}}},{{{}}},{}".format(test_key, parser_key, file)

    return (attr_tests, attr_parsers, attr_cmd_args)

def _resolve_test_related_attrs(ctx, attr_tests, attr_parsers, attr_cmd_args):
    """Resolves the format string passed as an argument to the bazel rule.
    It relies on the order of test and parser labels.
    """

    keys = {}
    for i, test in enumerate(attr_tests):
        test_key = "t{}".format(i)
        value = str(test.label)
        keys.update({test_key: value})

    for i, parser in enumerate(attr_parsers):
        parser_key = "p{}".format(i)
        parser_files = [x for x in parser.files.to_list() if x.is_source]
        if len(parser_files) > 1:
            fail("Parsers passed as attributes should be a single executable")
        [parser_file] = parser_files
        keys.update({parser_key: parser_file.short_path})

    return attr_cmd_args.format(**keys)

def _create_command(ctx):
    """Creates command for the script executed when invoking bazel run for
    a dashboard generation"""

    return 'python {dashboard} -r {root_dir} -w {working_dir} -t "{title}" {args} "$@"'.format(
        dashboard = ctx.executable._dashboard.short_path,
        root_dir = "$BUILD_WORKSPACE_DIRECTORY",
        working_dir = "$BUILD_WORKING_DIRECTORY",
        title = ctx.attr.title,
        args = _resolve_test_related_attrs(
            ctx,
            ctx.attr.tests,
            ctx.attr.parsers,
            ctx.attr.cmd_args,
        ),
    )

def _collect_transitive_files(ctx):
    """Collects transitive dependencies that are required for the run executable"""

    py_toolchain = ctx.toolchains["@bazel_tools//tools/python:toolchain_type"].py3_runtime
    return depset(
        direct = [py_toolchain.interpreter],
        transitive = [
                         dep[PyInfo].transitive_sources
                         for dep in ctx.attr.parsers
                         if PyInfo in dep
                     ] +
                     [ctx.attr._dashboard[PyInfo].transitive_sources] +
                     [py_toolchain.files],
    )

def _generate_dashboard_impl(ctx):
    """Implementation of the dashboard generation rule"""

    executable = ctx.actions.declare_file("{}_run.sh".format(ctx.attr.name))
    command = _create_command(ctx)

    ctx.actions.write(output = executable, content = command)
    py_toolchain = ctx.toolchains["@bazel_tools//tools/python:toolchain_type"].py3_runtime

    return [
        DefaultInfo(
            executable = executable,
            files = depset(ctx.files.tests),
            runfiles = ctx.runfiles(
                files = ctx.files.parsers,
                transitive_files = _collect_transitive_files(ctx),
            ).merge_all(
                [ctx.attr._dashboard.default_runfiles] +
                [dep.default_runfiles for dep in ctx.attr.parsers],
            ),
        ),
    ]

_generate_dashboard_attrs = {
    "tests": attr.label_list(
        doc = "Tests that will be run to generate the dashboard",
    ),
    "parsers": attr.label_list(
        doc = "Parsers to run on test output",
    ),
    "cmd_args": attr.string(
        doc = "Command for the dashboard generation tool",
    ),
    "title": attr.string(
        doc = "Title of the dashboard",
    ),
    "_dashboard": attr.label(
        cfg = "exec",
        executable = True,
        doc = "Script for running tests and parsing their outputs",
        default = Label("//xls/tools/dashboard:dashboard"),
    ),
}

generate_dashboard = rule(
    implementation = _generate_dashboard_impl,
    attrs = _generate_dashboard_attrs,
    executable = True,
    toolchains = ["@bazel_tools//tools/python:toolchain_type"],
)

def dashboard(name, title, tests):
    """Macro used to provide clear interface for the user. It converts the
    information provided by the user to the attributes understood by the
    underlying dashboard generation rule.
    """

    attr_tests, attr_parsers, attr_cmd_args = _prepare_test_related_attrs(tests)
    generate_dashboard(
        name = name,
        title = title,
        tests = attr_tests,
        parsers = attr_parsers,
        cmd_args = attr_cmd_args,
        testonly = True,
    )
