# Copyright 2025 The XLS Authors
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

"""XLS Diff-test with a target for updating the golden file."""

load("@bazel_skylib//rules:diff_test.bzl", _base_diff_test = "diff_test")

def _update_golden_impl(ctx):
    update_bin = ctx.actions.declare_file(ctx.label.name + "-binary.sh")
    ctx.actions.write(
        output = update_bin,
        content = r"""#!/usr/bin/env bash

set -xeuo pipefail

SRC="{file}"
DST="$BUILD_WORKING_DIRECTORY/{gold}"
cp $SRC $DST
""".format(
            file = ctx.file.new_golden.path[len(ctx.file.new_golden.root.path) + 1:],
            gold = ctx.label.workspace_root + "/" + ctx.label.package + "/" + ctx.attr.golden,
        ),
        is_executable = True,
    )
    return DefaultInfo(
        executable = update_bin,
        files = depset(direct = [update_bin]),
        runfiles = ctx.runfiles(files = [update_bin, ctx.file.new_golden]),
    )

_xls_update_golden = rule(
    implementation = _update_golden_impl,
    attrs = {
        "new_golden": attr.label(
            doc = "Label of the file being compared to the golden file.",
            allow_single_file = True,
            mandatory = True,
        ),
        "golden": attr.string(
            doc = "File where the golden file is. Must be relative to the rule's directory",
            mandatory = True,
        ),
    },
    executable = True,
)

def diff_test(name, file, golden, failure_message = None, tags = [], **kwargs):
    _base_diff_test(
        name = name,
        file1 = file,
        file2 = golden,
        failure_message = failure_message,
        tags = tags,
        **kwargs
    )
    _xls_update_golden(
        name = name + "_update_golden",
        new_golden = file,
        golden = golden,
        tags = tags + ["local", "manual"],
    )
