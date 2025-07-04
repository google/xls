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

load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")
load(
    "//xls/build_rules:xls_providers.bzl",
    "XlsOptimizationPassRegistryInfo",
)

def _generate_documentation_impl(ctx):
    passes = ctx.attr._passes[XlsOptimizationPassRegistryInfo].pass_infos
    protos = []
    cc_toolchain = find_cpp_toolchain(ctx)
    i = 0
    for p in passes:
        i += 1
        ccinfo = p.pass_impl
        out = ctx.actions.declare_file("{}_{}_{}.doc.binpb".format(
            ctx.attr.name,
            i,
            ccinfo.compilation_context.direct_public_headers[0].basename,
        ))
        args = ctx.actions.args()

        # TODO(allight): Unfortunate to check only the first header.
        args.add("-header", ccinfo.compilation_context.direct_public_headers[0].path)
        if ctx.attr.strip_prefix:
            args.add("-strip_prefix", ctx.attr.strip_prefix)
        args.add("-output", out.path)
        copts = []

        # TODO(allight): It would be nice if there was some way to just get c-flags from bzl.
        for define in ccinfo.compilation_context.defines.to_list():
            copts.append("-D{}".format(define))
        for ext_incl in ccinfo.compilation_context.external_includes.to_list():
            copts.append("-isystem{}".format(ext_incl))
        for frm in ccinfo.compilation_context.framework_includes.to_list():
            copts.append("-F{}".format(frm))
        for inc in ccinfo.compilation_context.includes.to_list():
            copts.append("-I{}".format(inc))
        for ld in ccinfo.compilation_context.local_defines.to_list():
            copts.append("-D{}".format(ld))
        for inc in ccinfo.compilation_context.quote_includes.to_list():
            copts.append("-iquote{}".format(inc))
        for ext_incl in ccinfo.compilation_context.system_includes.to_list():
            copts.append("-isystem{}".format(ext_incl))
        for builtin in cc_toolchain.built_in_include_directories:
            copts.append("-isystem{}".format(builtin))
        copts.append("--sysroot={}".format(cc_toolchain.sysroot))
        copts.extend(ctx.fragments.cpp.copts)
        copts.extend(ctx.fragments.cpp.cxxopts)
        args.add_joined("-copts", copts, join_with = ",")
        ctx.actions.run(
            outputs = [out],
            executable = ctx.executable._generate_documentation_proto,
            inputs = depset(transitive = [ccinfo.compilation_context.headers, cc_toolchain.all_files]),
            arguments = [args],
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
            default = "//xls/passes:passes",
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
