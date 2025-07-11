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

"""Helpers to generate pass documentation."""

load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")
load(
    "//xls/build_rules:xls_providers.bzl",
    "XlsOptimizationPassRegistryInfo",
)

def _cflags_for_info(comp_ctx, cc_toolchain, cc_feature):
    """Returns the copts and environment variables for compiling the given compilation context."""
    MARKER_SRC_FILE = "marker-src-file"
    comp_variables = cc_common.create_compile_variables(
        feature_configuration = cc_feature,
        cc_toolchain = cc_toolchain,
        # We pass the source file manually. This is removed from the flags list.
        source_file = MARKER_SRC_FILE,
        # We don't actually care about the output.
        output_file = "/dev/null",
        include_directories = comp_ctx.includes,
        quote_include_directories = comp_ctx.quote_includes,
        system_include_directories = comp_ctx.system_includes,
        preprocessor_defines = depset(transitive = [comp_ctx.defines, comp_ctx.local_defines]),
    )
    copts = cc_common.get_memory_inefficient_command_line(
        feature_configuration = cc_feature,
        action_name = ACTION_NAMES.cpp_compile,
        variables = comp_variables,
    )
    cenv = cc_common.get_environment_variables(
        feature_configuration = cc_feature,
        action_name = ACTION_NAMES.cpp_compile,
        variables = comp_variables,
    )
    copts = copts[:]

    # Remove the file and output arg.
    o_idx = copts.index("-o")
    copts.pop(o_idx)
    copts.pop(o_idx)
    copts.pop(copts.index(MARKER_SRC_FILE))

    return copts, cenv

def _generate_documentation_impl(ctx):
    passes = ctx.attr._passes[XlsOptimizationPassRegistryInfo].pass_infos
    protos = []
    cc_toolchain = find_cpp_toolchain(ctx)
    cc_feature = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        # TODO(allight): disabling module-maps is needed though I'm not really sure why.
        unsupported_features = ctx.disabled_features + ["module_maps"],
    )
    i = 0
    for p in passes:
        i += 1
        ccinfo = p.pass_impl

        out = ctx.actions.declare_file("{}_{}_{}.doc.binpb".format(
            ctx.attr.name,
            i,
            ccinfo.compilation_context.direct_public_headers[0].basename,
        ))

        copts, cenv = _cflags_for_info(ccinfo.compilation_context, cc_toolchain, cc_feature)

        args = ctx.actions.args()

        # TODO(allight): Unfortunate to check only the first header.
        args.add("-header", ccinfo.compilation_context.direct_public_headers[0].path)
        if ctx.attr.strip_prefix:
            args.add("-strip_prefix", ctx.attr.strip_prefix)
        args.add("-output", out.path)
        copts = [x.replace(",", "__ESCAPED_COMMA__") for x in copts]
        args.add_joined("-copts", copts, join_with = ",")

        ctx.actions.run(
            outputs = [out],
            executable = ctx.executable._generate_documentation_proto,
            inputs = depset(transitive = [ccinfo.compilation_context.headers, cc_toolchain.all_files]),
            arguments = [args],
            env = cenv,
            mnemonic = "ExtractPassComments",
            progress_message = "Parsing pass comment for %s" % ccinfo.compilation_context.direct_public_headers[0].path,
        )
        protos.append(out)

    mdfile = ctx.actions.declare_file("passes_documentation." + ctx.attr.name + ".md")
    args = ctx.actions.args()
    args.add("-pipeline", ctx.attr._passes[XlsOptimizationPassRegistryInfo].pipeline_binpb.path)
    args.add("-output", mdfile.path)
    args.add("-link_format", ctx.attr.codelink_format)
    for p in protos:
        args.add("-passes", p.path)
    ctx.actions.run(
        outputs = [mdfile],
        executable = ctx.executable._generate_documentation_md,
        inputs = protos + [ctx.attr._passes[XlsOptimizationPassRegistryInfo].pipeline_binpb],
        arguments = [args],
        mnemonic = "GenerateMarkdownPasses",
        progress_message = "Generating pass list markdown",
    )

    return [
        DefaultInfo(
            files = depset(direct = [mdfile]),
            data_runfiles = ctx.runfiles(
                files = [mdfile],
            ),
        ),
    ]

xls_generate_documentation = rule(
    implementation = _generate_documentation_impl,
    attrs = {
        "_generate_documentation_proto": attr.label(
            default = "//xls/passes/tools:generate_documentation_proto_main",
            executable = True,
            cfg = "exec",
            allow_files = True,
        ),
        "_generate_documentation_md": attr.label(
            default = "//xls/passes/tools:generate_documentation_md",
            executable = True,
            cfg = "exec",
        ),
        "_passes": attr.label(
            default = "//xls/passes:oss_optimization_passes",
            providers = [[XlsOptimizationPassRegistryInfo]],
        ),
        "strip_prefix": attr.string(
            doc = "string to strip out for file names",
            default = "",
            mandatory = False,
        ),
        "codelink_format": attr.string(
            doc = "String with '%s' where the (stripped) filename should be inserted to create a link to that file's source code.",
            mandatory = True,
        ),
    },
    fragments = ["cpp"],
    toolchains = use_cpp_toolchain(),
)
