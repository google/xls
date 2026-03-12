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
load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load(
    "//xls/build_rules:xls_common_rules.bzl",
    "get_input_infos",
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
load(
    "//xls/build_rules:xls_utilities.bzl",
    "BoolConfigSettingInfo",
)

_PROTO_FILE_EXTENSION = ".pb"

_OBJ_FILE_EXTENSION = ".o"

visibility(["//xls/build_rules/...", "//xls/jit"])

_xls_aot_files_attrs = {
    "with_msan": attr.bool(
        doc = "if the jit code should be compiled with msan",
        mandatory = True,
    ),
    "llvm_opt_level": attr.int(
        doc = "What opt level to compile aot files with",
        default = 3,
        mandatory = False,
    ),
    "aot_target": attr.string(
        doc = "What target to generate aot obj files for",
        default = "native",
        mandatory = False,
    ),
    "salt_symbols": attr.bool(
        default = True,
        doc = "Use target label to uniqify the symbol names.",
    ),
    "_save_temps_is_requested": attr.label(
        doc = "save_temps config",
        default = "//xls/common/config:save_temps_is_requested",
        providers = [BoolConfigSettingInfo],
    ),
    "_jit_emulated_tls": attr.label(
        doc = "emulated tls implementation",
        default = "//xls/jit:jit_emulated_tls",
        providers = [CcInfo],
    ),
    "_emit_aot_intermediates": attr.label(
        doc = "emit intermediates",
        default = "//xls/common/config:emit_aot_intermediates",
    ),
    "top_type": attr.string(
        doc = "Type of top (FUNCTION/PROC/BLOCK)",
        default = "FUNCTION",
        mandatory = False,
    ),
    "jobs": attr.int(
        doc = "Number of jobs to use for compilation. If not 1 then compilation will be done " +
              "in by splitting the design into multiple pieces which are optimized and comipled " +
              "into object code separately and then linked.",
        default = 1,
        mandatory = False,
    ),
    "enable_llvm_coverage": attr.bool(
        doc = "If true, passes --enable_llvm_coverage to the AOT compiler to instrument the " +
              "generated code for LLVM coverage.",
        default = False,
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
    jobs = ctx.attr.jobs
    src = get_src_ir_for_xls(ctx)
    aot_compiler = ctx.executable._xls_aot_compiler_tool
    out_proto_filename = ctx.attr.name + _PROTO_FILE_EXTENSION
    proto_file = ctx.actions.declare_file(out_proto_filename)
    args = ctx.actions.args()
    skeleton_args = ctx.actions.args()

    def common_add(*va, **kwargs):
        args.add(*va, **kwargs)
        skeleton_args.add(*va, **kwargs)

    common_add("-input", src.ir_file.path)
    if (ctx.attr.salt_symbols):
        common_add("-symbol_salt", str(ctx.label))
    common_add("-aot_target", ctx.attr.aot_target)
    common_add("-top", ctx.attr.top)
    common_add("-top_type", ctx.attr.top_type)
    other_linking_contexts = []
    if ctx.attr.with_msan:
        common_add("--include_msan=true")

        # With msan we need the TLS implementation.
        other_linking_contexts = [ctx.attr._jit_emulated_tls[CcInfo].linking_context]
    else:
        common_add("--include_msan=false")

    if ctx.attr.enable_llvm_coverage:
        common_add("--enable_llvm_coverage=true")
    else:
        common_add("--enable_llvm_coverage=false")

    extra_files = []

    skeleton_args.add("-output_proto", proto_file.path)

    # NB If we have jobs we don't really need to do anything to get save_temps to work since the
    # split llvm ir files will be saved anyway. Note that the asm and opt_ir won't be saved however.
    # TODO(allight): It would be nice to emit the opt_ir and asm in this situation as well.
    if jobs == 1:
        aot_direct_request = ctx.attr._emit_aot_intermediates[BuildSettingInfo].value
        save_temps_request = ctx.attr._save_temps_is_requested[BoolConfigSettingInfo].value
        if aot_direct_request or save_temps_request:
            textproto_file = ctx.actions.declare_file(ctx.attr.name + ".text_proto")
            llvm_ir_file = ctx.actions.declare_file(ctx.attr.name + ".ll")
            llvm_opt_ir_file = ctx.actions.declare_file(ctx.attr.name + ".opt.ll")
            asm_file = ctx.actions.declare_file(ctx.attr.name + ".S")
            extra_files += [llvm_ir_file, llvm_opt_ir_file, asm_file, textproto_file]
            args.add("-output_textproto", textproto_file.path)
            args.add("-output_llvm_ir", llvm_ir_file.path)
            args.add("-output_llvm_opt_ir", llvm_opt_ir_file.path)
            args.add("-output_asm", asm_file.path)
        out_obj_filename = ctx.attr.name + _OBJ_FILE_EXTENSION
        obj_file = ctx.actions.declare_file(out_obj_filename)
        args.add("-llvm_opt_level", ctx.attr.llvm_opt_level)

        args.add("-output_object", obj_file.path)

        # Non-skeleton run to create the object file.
        ctx.actions.run(
            outputs = [obj_file] + extra_files,
            inputs = [src.ir_file],
            arguments = [args],
            executable = aot_compiler,
            mnemonic = "AOTCompiling",
            progress_message = "Aot(JIT)Compiling %{label}: %{input}",
            toolchain = None,
            execution_requirements = {tag: "" for tag in ctx.attr.tags},
        )
        obj_files = [obj_file]
        obj_file_outputs = cc_common.create_compilation_outputs(
            objects = depset([obj_file]),
            pic_objects = depset([obj_file]),
        )
    else:
        unopt_llvm_ir_file = ctx.actions.declare_file(ctx.attr.name + ".bc")
        split_files = []
        for i in range(jobs):
            split_file = ctx.actions.declare_file(ctx.attr.name + ".part." + str(i) + ".bc")
            extra_files.append(split_file)
            split_files.append(split_file)
        args.add("-output_llvm_ir", unopt_llvm_ir_file.path)

        # Non-skeleton run to create the unoptimized llvm ir file.
        ctx.actions.run(
            outputs = [unopt_llvm_ir_file],
            inputs = [src.ir_file],
            arguments = [args],
            executable = aot_compiler,
            mnemonic = "AOTCompiling",
            progress_message = "Aot(JIT)Compiling %{label}: %{input}",
            toolchain = None,
            execution_requirements = {tag: "" for tag in ctx.attr.tags},
        )
        ctx.actions.run(
            outputs = split_files,
            inputs = [unopt_llvm_ir_file],
            arguments = [
                ctx.actions.args()
                    .add("-input", unopt_llvm_ir_file.path)
                    .add("-outputs", ",".join([f.path for f in split_files]))
                    .add("-private_salt", str(ctx.label)),
            ],
            executable = ctx.executable._xls_aot_generate_compiler_segments_tool,
            mnemonic = "AOTGenerateCompilerSegments",
            progress_message = "AOTGenerateCompilerSegments %{label}: %{input}",
            toolchain = None,
            execution_requirements = {tag: "" for tag in ctx.attr.tags},
        )
        obj_files = [
            ctx.actions.declare_file(ctx.attr.name + ".part." + str(i) + ".o")
            for i in range(jobs)
        ]
        for i in range(jobs):
            piece_args = ctx.actions.args()
            piece_args.add("-input", split_files[i].path)
            piece_args.add("-output_object", obj_files[i].path)
            piece_args.add("-llvm_opt_level", ctx.attr.llvm_opt_level)
            piece_args.add("-aot_target", ctx.attr.aot_target)
            if ctx.attr.with_msan:
                piece_args.add("--include_msan=true")
            else:
                piece_args.add("--include_msan=false")
            if ctx.attr.enable_llvm_coverage:
                piece_args.add("--enable_llvm_coverage=true")
            else:
                piece_args.add("--enable_llvm_coverage=false")

            ctx.actions.run(
                outputs = [obj_files[i]],
                inputs = [split_files[i]],
                arguments = [piece_args],
                executable = ctx.executable._xls_aot_compiler_segment_tool,
                mnemonic = "AOTCompiling",
                progress_message = "AOTJitCompiling %{label}: %{input}",
                toolchain = None,
                execution_requirements = {tag: "" for tag in ctx.attr.tags},
            )
        obj_file_outputs = cc_common.create_compilation_outputs(
            objects = depset(obj_files),
            pic_objects = depset(obj_files),
        )

    # Skeleton run to create the proto file.
    ctx.actions.run(
        outputs = [proto_file],
        inputs = [src.ir_file],
        arguments = [skeleton_args],
        executable = aot_compiler,
        mnemonic = "AotSkeleton",
        progress_message = "Generating AOT skeleton %{label}: %{input}",
        toolchain = None,
        execution_requirements = {tag: "" for tag in ctx.attr.tags},
    )

    (linking_context, _linking_outputs) = cc_common.create_linking_context_from_compilation_outputs(
        name = ctx.label.name,
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        compilation_outputs = obj_file_outputs,
        linking_contexts = other_linking_contexts,
    )

    return [
        AotCompileInfo(object_files = obj_files, proto_file = proto_file),
        DefaultInfo(files = depset(obj_files + [proto_file] + extra_files)),
        CcInfo(linking_context = linking_context),
    ] + get_input_infos(ctx.attr.src)

xls_aot_generate = rule(
    implementation = _xls_aot_generate_impl,
    attrs = dicts.add(
        xls_ir_common_attrs,
        xls_ir_top_attrs,
        _xls_aot_files_attrs,
        CONFIG["xls_outs_attrs"],
        dicts.pick(xls_toolchain_attrs, [
            "_xls_aot_compiler_tool",
            "_xls_aot_compiler_segment_tool",
            "_xls_aot_generate_compiler_segments_tool",
        ]),
    ),
    fragments = ["cpp"],
    toolchains = use_cpp_toolchain(),
)
