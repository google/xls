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
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
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
    passes = ctx.attr.passes[XlsOptimizationPassRegistryInfo].pass_infos
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
            executable = ctx.executable.generate_documentation_proto,
            inputs = depset(transitive = [ccinfo.compilation_context.headers, cc_toolchain.all_files]),
            arguments = [args],
            env = cenv,
            mnemonic = "ExtractPassComments",
            progress_message = "Parsing pass comment for %s" % ccinfo.compilation_context.direct_public_headers[0].path,
            toolchain = None,
        )
        protos.append(out)

    mdfile = ctx.actions.declare_file("passes_documentation." + ctx.attr.name + ".md")
    args = ctx.actions.args()
    args.add("-pipeline", ctx.attr.passes[XlsOptimizationPassRegistryInfo].pipeline_binpb.path)
    args.add("-pipeline_txtpb_file", ctx.attr.passes[XlsOptimizationPassRegistryInfo].pipeline_src.path)
    args.add("-output", mdfile.path)
    if ctx.attr.strip_prefix:
        args.add("-strip_prefix", ctx.attr.strip_prefix)
    args.add("-link_format", ctx.attr.codelink_format)
    args.add("-jinja_template", ctx.file.template.path)
    for p in protos:
        args.add("-passes", p.path)
    ctx.actions.run(
        outputs = [mdfile],
        executable = ctx.executable._generate_documentation_md,
        inputs = protos + [ctx.attr.passes[XlsOptimizationPassRegistryInfo].pipeline_binpb, ctx.file.template],
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

_internal_xls_generate_documentation = rule(
    implementation = _generate_documentation_impl,
    attrs = {
        "_generate_documentation_md": attr.label(
            default = "//xls/passes/tools:generate_documentation_md",
            executable = True,
            cfg = "exec",
        ),
        "generate_documentation_proto": attr.label(
            executable = True,
            cfg = "exec",
        ),
        "passes": attr.label(
            cfg = "exec",
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
        "template": attr.label(
            allow_single_file = True,
            doc = "Jinja template for the passes_list.md file.",
        ),
    },
    fragments = ["cpp"],
    toolchains = use_cpp_toolchain(),
)

def _create_generate_documentation_proto_bin_impl(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)
    cc_feature = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

    # Link the binary for extracting the passes.
    generate_documentation_proto_linked = cc_common.link(
        actions = ctx.actions,
        name = ctx.attr.name,
        cc_toolchain = cc_toolchain,
        feature_configuration = cc_feature,
        linking_contexts = [
            ctx.attr._generate_documentation_proto[CcInfo].linking_context,
            ctx.attr.passes[XlsOptimizationPassRegistryInfo].cc_library.linking_context,
        ],
    )
    return DefaultInfo(
        files = depset(direct = [generate_documentation_proto_linked.executable]),
        data_runfiles = ctx.attr._generate_documentation_proto[DefaultInfo].data_runfiles,
        default_runfiles = ctx.attr._generate_documentation_proto[DefaultInfo].default_runfiles,
        executable = generate_documentation_proto_linked.executable,
    )

_create_generate_documentation_proto_bin = rule(
    implementation = _create_generate_documentation_proto_bin_impl,
    attrs = {
        "_generate_documentation_proto": attr.label(
            default = "//xls/passes/tools:generate_documentation_proto_main",
            cfg = "exec",
            providers = [[CcInfo]],
        ),
        "passes": attr.label(
            cfg = "exec",
            providers = [[XlsOptimizationPassRegistryInfo]],
            doc = "The pass registry to pull passes and compound passes from.",
        ),
    },
    fragments = ["cpp"],
    toolchains = use_cpp_toolchain(),
    doc = "Helper rule to link in the passes given by the attr.",
)

def xls_generate_documentation(
        name,
        passes,
        codelink_format,
        template,
        strip_prefix = "",
        tags = [],
        **kwargs):
    """Generates documentation for the given passes.

    Args:
      name: Name of the rule.
      passes: Target containing the passes to generate documentation for.
      codelink_format: String with '%s' where the (stripped) filename should be inserted to create a link to that file's source code.
      template: The file to use as the jinja template for the generated list.
      strip_prefix: String to strip from the beginning of file names.
      tags: Tags to apply to the rule.
      **kwargs: Additional arguments to pass to the underlying rule.
    """

    # TODO(allight): It's unfortunate we use a macro here. I wasn't able to
    # figure out how to make 'link'/actions.run correctly set up runfiles.
    _create_generate_documentation_proto_bin(
        name = "generate_documentation_proto_bin_for_%s" % name,
        passes = passes,
        tags = tags,
    )
    _internal_xls_generate_documentation(
        name = name,
        passes = passes,
        codelink_format = codelink_format,
        strip_prefix = strip_prefix,
        generate_documentation_proto = ":generate_documentation_proto_bin_for_%s" % name,
        template = template,
        tags = tags,
        **kwargs
    )
