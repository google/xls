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

"""This module contains rules not dependent on the XLS framework."""

def _check_sha256sum_test_impl(ctx):
    """The implementation of the 'check_sha256sum_test' rule.

    Validates the sha256sum checksum of a source file with a user-defined golden
    result.

    Args:
      ctx: The current rule's context object.

    Returns:
      DefaultInfo provider
    """
    src = ctx.file.src
    runfiles = [src]
    executable_file = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(
        output = executable_file,
        content = "\n".join([
            "#!/bin/bash",
            "set -e",
            # Note two spaces is required between the sha256sum and filename to
            # comply with the sha256sum input format.
            "echo \"{}  {}\" | sha256sum -c -".format(
                ctx.attr.sha256sum,
                src.short_path,
            ),
            "if [ $? -ne 0 ]; then",
            "echo \"Error: sha256sum checksum mismatch for file '{}'.\""
                .format(src.short_path),
            "exit -1",
            "fi",
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

_check_sha256sum_test_attrs = {
    "src": attr.label(
        doc = "The source file.",
        allow_single_file = True,
        mandatory = True,
    ),
    "sha256sum": attr.string(
        doc = "The sha256sum of the source file.",
        mandatory = True,
    ),
}

check_sha256sum_test = rule(
    doc = """Validates the sha256sum checksum of a source file with a user-defined golden result.

        Example:
        ```
            check_sha256sum_test(
                name = "generated_file_sha256sum_test",
                src = ":generated_file.x",
                sha256sum = "6522799f7b64dbbb2a31eb2862052b8988e78821d8b61fff7f508237a9d9f01d",
            )
        ```
    """,
    implementation = _check_sha256sum_test_impl,
    attrs = _check_sha256sum_test_attrs,
    test = True,
)
