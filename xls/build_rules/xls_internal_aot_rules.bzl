# Copyright 2024 The XLS Authors
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
This module contains rules for generating AOT compiled ir entrypoints.
"""

load("@bazel_skylib//lib:dicts.bzl", "dicts")
load(
    "//xls/build_rules:xls_common_rules.bzl",
    "get_src_ir_for_xls",
)
load(
    "//xls/build_rules:xls_config_rules.bzl",
    "CONFIG",
)
load(
    "//xls/build_rules:xls_ir_rules.bzl",
    "xls_ir_common_attrs",
    "xls_ir_top_attrs",
)
load("//xls/build_rules:xls_providers.bzl", "AotCompileInfo")
load(
    "//xls/build_rules:xls_toolchains.bzl",
    "xls_toolchain_attrs",
)
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")

_PROTO_FILE_EXTENSION = ".pb"

_OBJ_FILE_EXTENSION = ".o"

visibility(["//xls/build_rules", "//xls/jit"])

_xls_aot_files_attrs = {
    "with_msan": attr.bool(
        doc = "if the jit code should be compiled with msan",
        mandatory = True,
    ),
    "_jit_emulated_tls": attr.label(
        doc = "emulated tls implementation",
        default = "//xls/jit:jit_emulated_tls",
        providers = [CcInfo],
    ),
}

def _xls_aot_generate_impl(ctx):
    """Generate an aot object file and a proto describing it."""
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    src = get_src_ir_for_xls(ctx)
    aot_compiler = ctx.executable._xls_aot_compiler_tool
    out_proto_filename = ctx.attr.name + _PROTO_FILE_EXTENSION
    out_obj_filename = ctx.attr.name + _OBJ_FILE_EXTENSION
    proto_file = ctx.actions.declare_file(out_proto_filename)
    obj_file = ctx.actions.declare_file(out_obj_filename)
    args = ctx.actions.args()
    args.add("-input", src.path)
    args.add("-top", ctx.attr.top)
    args.add("-output_object", obj_file.path)
    args.add("-output_proto", proto_file.path)
    other_linking_contexts = []
    if ctx.attr.with_msan:
        args.add("--include_msan=true")

        # With msan we need the TLS implementation.
        other_linking_contexts = [ctx.attr._jit_emulated_tls[CcInfo].linking_context]
    else:
        args.add("--include_msan=false")
    ctx.actions.run(
        outputs = [proto_file, obj_file],
        inputs = [src],
        arguments = [args],
        executable = aot_compiler,
        mnemonic = "AOTCompiling",
        progress_message = "Aot(JIT)Compiling %{label}: %{input}",
        toolchain = None,
    )
    obj_file_outputs = cc_common.create_compilation_outputs(
        objects = depset([obj_file]),
        pic_objects = depset([obj_file]),
    )
    (linking_context, _linking_outputs) = cc_common.create_linking_context_from_compilation_outputs(
        name = ctx.label.name,
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        compilation_outputs = obj_file_outputs,
        linking_contexts = other_linking_contexts,
    )
    cc_common.merge_compilation_contexts()

    return [
        AotCompileInfo(object_file = obj_file, proto_file = proto_file),
        DefaultInfo(files = depset([obj_file, proto_file])),
        CcInfo(linking_context = linking_context),
    ]

xls_aot_generate = rule(
    implementation = _xls_aot_generate_impl,
    attrs = dicts.add(
        xls_ir_common_attrs,
        xls_ir_top_attrs,
        _xls_aot_files_attrs,
        CONFIG["xls_outs_attrs"],
        xls_toolchain_attrs,
    ),
    fragments = ["cpp"],
    toolchains = use_cpp_toolchain(),
)
