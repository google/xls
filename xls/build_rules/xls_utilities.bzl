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

"""This module contains helper rules."""

load("@rules_proto//proto:defs.bzl", "ProtoInfo")
load(
    "//xls/build_rules:xls_common_rules.bzl",
    "get_output_filename_value",
)

_PROTOBIN_FILE_EXTENSION = ".protobin"

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
            "#!/usr/bin/env bash",
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

Examples:

1. A simple example.

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
        # The files required for generating the frozen file.
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
    return [DefaultInfo(files = depset(generated_files))]

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

```
check_sha256sum_frozen(
    name = "my_output_checksum",
    src = ":my_output.v",
    sha256sum = "d1bc8d3ba4afc7e109612cb73acbdddac052c93025aa1f82942edabb7deb82a1",
    frozen_file = "my_output.frozen.x",
)
```

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

Examples:

1. A simple example.

    ```
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
    ```
    """,
    implementation = _check_sha256sum_frozen_impl,
    attrs = _check_sha256sum_frozen_attrs,
)

def _proto_data_impl(ctx):
    """The implementation of the 'proto_data' rule.

    Args:
      ctx: The current rule's context object.

    Returns:
      DefaultInfo provider
    """
    src = ctx.file.src
    proto2bin_tool = ctx.executable._proto2bin_tool
    protobin_filename = get_output_filename_value(
        ctx,
        "protobin_file",
        ctx.attr.name + _PROTOBIN_FILE_EXTENSION,
    )
    protobin_file = ctx.actions.declare_file(protobin_filename)
    ctx.actions.run_shell(
        outputs = [protobin_file],
        # The files required for generating the protobin file.
        inputs = [src, proto2bin_tool],
        # The proto2bin executable is a tool needed by the action.
        tools = [proto2bin_tool],
        command = "{} {} --message {} --output {}".format(
            proto2bin_tool.path,
            src.path,
            ctx.attr.proto_name,
            protobin_file.path,
        ),
        use_default_shell_env = True,
        mnemonic = "Proto2Bin",
        progress_message = "Checking sha256sum on file: %s" % (src.path),
    )
    return [DefaultInfo(files = depset([protobin_file]))]

_proto_data_attrs = {
    "src": attr.label(
        doc = "The source file.",
        allow_single_file = True,
        mandatory = True,
    ),
    "protobin_file": attr.output(
        doc = "The name of the output file to write binary proto to. If not " +
              "specified, the target name of the bazel rule followed by a " +
              _PROTOBIN_FILE_EXTENSION + " extension is used.",
    ),
    "proto_name": attr.string(
        doc = "The name of the message type in the .proto files that 'src' " +
              "file represents.",
        default = "xlscc.HLSBlock",
    ),
    "_proto2bin_tool": attr.label(
        doc = "Convert a proto text to a proto binary.",
        default = Label("//xls/tools:proto2bin"),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
}

proto_data = rule(
    doc = """Converts a proto text with a xlscc.HLSBlock message to a proto binary.

This rules is used in conjunction with the (e.g. xls_cc_ir and xls_cc_verilog)
rules and xls_cc_* (e.g. xls_cc_ir_macro and xls_cc_verilog_macro) macros.

Examples:

1. A simple example.

    ```
    proto_data(
        name = "packet_selector_block_pb",
        src = "packet_selector.textproto",
    )
    ```
    """,
    implementation = _proto_data_impl,
    attrs = _proto_data_attrs,
)

def _proto_to_dslx_impl(ctx):
    """The implementation of the 'proto_to_dslx' rule.

    Args:
      ctx: The current rule's context object.

    Returns:
      DefaultInfo provider
    """
    schema_sources = ctx.attr.proto_schema[ProtoInfo].direct_sources
    schema = schema_sources[0]

    proto2dslx = ctx.executable._proto_to_dslx_tool
    dslx_filename = get_output_filename_value(
        ctx,
        "dslx_file",
        ctx.attr.name + ".x",
    )
    dslx_file = ctx.actions.declare_file(dslx_filename)
    ctx.actions.run_shell(
        outputs = [dslx_file],
        # The files required for generating the protobin file.
        inputs = [schema, ctx.file.textproto, proto2dslx],
        # The proto2bin executable is a tool needed by the action.
        tools = [proto2dslx],
        command = "{} --proto_def_path {} --proto_name {} --textproto_path {} --var_name {} --output_path {}".format(
            proto2dslx.path,
            schema.path,
            ctx.attr.proto_name,
            ctx.file.textproto.path,
            ctx.attr.variable_name,
            dslx_file.path,
        ),
        use_default_shell_env = True,
        mnemonic = "Proto2DSLX",
    )
    return [DefaultInfo(files = depset([dslx_file]))]

_proto_to_dslx_attrs = {
    "proto_schema": attr.label(
        doc = "Schema definition proto file",
        providers = [ProtoInfo],
        mandatory = True,
    ),
    "proto_name": attr.string(
        doc = "The name of the message type in the .proto files that 'proto_def_path' " +
              "file represents.",
        mandatory = True,
    ),
    "textproto": attr.label(
        doc = "The source textproto file to convert to DSLX",
        allow_single_file = True,
        mandatory = True,
    ),
    "variable_name": attr.string(
        doc = "The name of the DSLX struct variable with contents of the textproto " +
              "file. Defaults to 'parameters'",
        default = "PARAMETERS",
    ),
    "dslx_file": attr.output(
        doc = "The name of the output file to write DSLX output to. " +
              "Defaults to <rule_name>.x",
    ),
    "_proto_to_dslx_tool": attr.label(
        doc = "Convert a proto text to a dslx",
        default = Label("//xls/tools:proto_to_dslx_main"),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
}

proto_to_dslx = rule(
    implementation = _proto_to_dslx_impl,
    attrs = _proto_to_dslx_attrs,
)
