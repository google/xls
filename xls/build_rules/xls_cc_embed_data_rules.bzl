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

"""
This module contains embedded data rules for xls.
"""

load("@bazel_skylib//lib:dicts.bzl", "dicts")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")

embed_data_attrs = {
    "_absl_span": attr.label(
        default = Label("@com_google_absl//absl/types:span"),
        providers = [CcInfo],
    ),
    "_embed_data": attr.label(
        default = Label("//xls/dev_tools/embed_data:create_source_files"),
        cfg = "exec",
        executable = True,
        allow_files = True,
    ),
}

_xls_cc_embed_data_attrs = dicts.add(
    {
        "data": attr.label(
            mandatory = True,
            allow_single_file = True,
            doc = "The data to embed.",
        ),
        "namespace": attr.string(
            default = "xls",
            doc = "The namespace to place the accessor in.",
        ),
    },
    embed_data_attrs,
)

def get_embedded_data(
        ctx,
        name,
        hdr_file,
        cpp_file,
        namespace,
        accessor,
        data_file):
    """Embeds the given data file into a C++ library.

    Args:
      ctx: The rule context.
      name: The name of the rule. This is used to create the intermediate
            library files and should be unique.
      hdr_file: The output header file.
      cpp_file: The output C++ file.
      namespace: The namespace to place the accessor in.
      accessor: The name of the accessor function.
      data_file: The data file to embed.

    Returns:
      A CcInfo provider.
    """
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    args = ctx.actions.args()
    args.add("-namespace", namespace)
    args.add("-data_file", data_file.path)
    args.add("-output_header", hdr_file.path)
    args.add("-output_source", cpp_file.path)
    args.add("-accessor", accessor)
    ctx.actions.run(
        executable = ctx.executable._embed_data,
        inputs = [data_file],
        outputs = [hdr_file, cpp_file],
        arguments = [args],
        mnemonic = "EmbedDataSources",
        progress_message = "Generating Embedded Data sources",
        toolchain = None,
        use_default_shell_env = True,
    )
    (comp_ctx, comp_outputs) = cc_common.compile(
        name = name,
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        srcs = [cpp_file],
        public_hdrs = [hdr_file],
        compilation_contexts = [ctx.attr._absl_span[CcInfo].compilation_context],
        additional_inputs = [data_file],
    )
    (link_ctx, _link_outputs) = cc_common.create_linking_context_from_compilation_outputs(
        name = name,
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        compilation_outputs = comp_outputs,
        linking_contexts = [ctx.attr._absl_span[CcInfo].linking_context],
    )
    return CcInfo(compilation_context = comp_ctx, linking_context = link_ctx)

def _xls_cc_embed_data_impl(ctx):
    hdr_filename = "%s_embedded.h" % ctx.attr.name
    cpp_filename = "%s_embedded.cc" % ctx.attr.name
    hdr_file = ctx.actions.declare_file(hdr_filename)
    cpp_file = ctx.actions.declare_file(cpp_filename)

    return [
        get_embedded_data(
            ctx = ctx,
            name = ctx.attr.name,
            hdr_file = hdr_file,
            cpp_file = cpp_file,
            namespace = ctx.attr.namespace,
            accessor = "get_%s" % ctx.attr.name,
            data_file = ctx.file.data,
        ),
        DefaultInfo(files = depset([hdr_file, cpp_file])),
    ]

xls_cc_embed_data = rule(
    implementation = _xls_cc_embed_data_impl,
    attrs = _xls_cc_embed_data_attrs,
    provides = [CcInfo],
    fragments = ["cpp"],
    toolchains = use_cpp_toolchain(),
    doc = "Embed binary data into a cc library. The data is retrieved from 'get_<name>()'",
)
