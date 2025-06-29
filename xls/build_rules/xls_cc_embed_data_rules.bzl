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

load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")

_xls_cc_embed_data_attrs = {
    "data": attr.label(
        mandatory = True,
        allow_single_file = True,
        doc = "The data to embed.",
    ),
    "namespace": attr.string(
        default = "xls",
        doc = "The namespace to place the accessor in.",
    ),
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

def _xls_cc_embed_data_impl(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    hdr_filename = "%s_embedded.h" % ctx.attr.name
    cpp_filename = "%s_embedded.cc" % ctx.attr.name
    hdr_file = ctx.actions.declare_file(hdr_filename)
    cpp_file = ctx.actions.declare_file(cpp_filename)
    args = ctx.actions.args()
    args.add("-namespace", ctx.attr.namespace)
    args.add("-data_file", ctx.file.data.path)
    args.add("-output_header", hdr_file.path)
    args.add("-output_source", cpp_file.path)
    args.add("-accessor", "get_%s" % ctx.attr.name)
    ctx.actions.run(
        executable = ctx.executable._embed_data,
        inputs = [ctx.file.data],
        outputs = [hdr_file, cpp_file],
        arguments = [args],
        mnemonic = "EmbedDataSources",
        progress_message = "Generating Embedded Data sources",
        toolchain = None,
    )
    (comp_ctx, comp_outputs) = cc_common.compile(
        name = ctx.label.name,
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        srcs = [cpp_file],
        public_hdrs = [hdr_file],
        compilation_contexts = [ctx.attr._absl_span[CcInfo].compilation_context],
        additional_inputs = [ctx.file.data],
    )
    (link_ctx, _link_outputs) = cc_common.create_linking_context_from_compilation_outputs(
        name = ctx.label.name,
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        compilation_outputs = comp_outputs,
        linking_contexts = [ctx.attr._absl_span[CcInfo].linking_context],
    )
    return [
        CcInfo(compilation_context = comp_ctx, linking_context = link_ctx),
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
