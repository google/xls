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
    doc = """Validates the sha256sum checksum of a source file with a user-defined checksum.

        This rule is typically used to ensure that the contents of a file is
        unchanged.

        Example:
            check_sha256sum_test(
                name = "generated_file_sha256sum_test",
                src = ":generated_file.x",
                sha256sum = "6522799f7b64dbbb2a31eb2862052b8988e78821d8b61fff7f508237a9d9f01d",
            )
    """,
    implementation = _check_sha256sum_test_impl,
    attrs = _check_sha256sum_test_attrs,
    test = True,
)

def _check_sha256sum_frozen_impl(ctx):
    """The implementation of the 'check_sha256sum_frozen' rule.

    Validates the sha256sum checksum of a source file with a user-defined golden
    result. If the validation result is a match between the sha256sum checksum
    of the source file with a user-defined golden result, then an output file is
    created with the same content as the source file.

    Args:
      ctx: The current rule's context object.

    Returns:
      DefaultInfo provider
    """
    src = ctx.file.src
    generated_files = []
    frozen_file = ctx.actions.declare_file(ctx.attr.frozen_file.name)
    generated_files.append(frozen_file)
    src_path = src.path
    ctx.actions.run_shell(
        outputs = [frozen_file],
        # The files required for converting the DSLX source file.
        inputs = [src],
        command = "\n".join([
            # Note two spaces is required between the sha256sum and filename to
            # comply with the sha256sum input format.
            "echo \"{}  {}\" | sha256sum -c -".format(
                ctx.attr.sha256sum,
                src_path,
            ),
            "if [ $? -ne 0 ]; then",
            "echo \"Error: sha256sum checksum mismatch for file '{}'.\""
                .format(src_path),
            "exit -1",
            "fi",
            "cat {} > {}".format(src_path, frozen_file.path),
            "exit 0",
        ]),
        use_default_shell_env = True,
        mnemonic = "CheckSHA256SumFrozen",
        progress_message = "Checking sha256sum on file: %s" % (src_path),
    )
    return [
        DefaultInfo(
            files = depset(generated_files),
        ),
    ]

_check_sha256sum_frozen_attrs = {
    "src": attr.label(
        doc = "The source file.",
        allow_single_file = True,
        mandatory = True,
    ),
    "sha256sum": attr.string(
        doc = "The sha256sum of the source file.",
        mandatory = True,
    ),
    "frozen_file": attr.output(
        doc = "The frozen output file.",
        mandatory = True,
    ),
}

check_sha256sum_frozen = rule(
    doc = """Produces a frozen file if the sha256sum checksum of a source file matches a user-defined checksum.

        As projects cut releases or freeze, it's important to know that
        generated (e.g. Verilog) code is never changing without having to
        actually check in the generated artifact. This rule performs a checksum
        of a generated file as an integrity check. Users might use this rule to
        help enable confidence that there is neither:

        *   non-determinism in the toolchain, nor
        *   an accidental dependence on a non-released toolchain (e.g. an
            accidental dependence on top-of-tree, where the toolchain is
            constantly changing)

        Say there was a codegen rule producing `my_output.v`, a user might instantiate
        something like:

            check_sha256sum_frozen(
                name = "my_output_checksum",
                src = ":my_output.v",
                sha256sum = "d1bc8d3ba4afc7e109612cb73acbdddac052c93025aa1f82942edabb7deb82a1",
                frozen_file = "my_output.frozen.x",
            )

        ... and then take a dependency on `my_output.frozen.v` in the
        surrounding project, knowing that it had been checksum-verified.

        Taking a dependence on `my_output.v` directly may also be ok if the
        `:my_output_checksum` target is also built (e.g. via the same wildcard
        build request), but taking a dependence on the output `.frozen.v` file
        ensures that the checking is an integral part of the downstream
        build-artifact-creation process.

        At its core, this rule ensure that the contents of a file does not
        change by verifying that it matches a given checksum. Typically, this
        rule is used to control the build process. The rule serves as a trigger
        on rules depending on its output (the frozen file). When the validation
        of the sha256sum succeed, rules depending on the frozen file are
        built/executed. When the validation of the sha256sum fails, rules
        depending on the frozen file are not built/executed.

        In the example below, when the validation of the sha256sum for
        target 'generated_file_sha256sum_frozen' succeeds, target
        'generated_file_dslx' is built. However, when the validation of the
        sha256sum for target 'generated_file_sha256sum_frozen' fails, target
        'generated_file_dslx' is not built.

        Example:
            check_sha256sum_frozen(
                name = "generated_file_sha256sum_frozen",
                src = ":generated_file.x",
                sha256sum = "6522799f7b64dbbb2a31eb2862052b8988e78821d8b61fff7f508237a9d9f01d",
                frozen_file = "generated_file.frozen.x",
            )

            dslx_library(
                name = "generated_file_dslx",
                src = ":generated_file.frozen.x",
            )
    """,
    implementation = _check_sha256sum_frozen_impl,
    attrs = _check_sha256sum_frozen_attrs,
)
