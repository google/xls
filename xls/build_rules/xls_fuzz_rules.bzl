# Copyright 2026 The XLS Authors
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

"""This module contains the dslx_fuzz_test rule."""

load("@bazel_skylib//lib:dicts.bzl", "dicts")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load(
    "//xls/build_rules:xls_internal_aot_rules.bzl",
    "xls_aot_compile",
    "xls_aot_env_attrs",
)
load(
    "//xls/build_rules:xls_ir_rules.bzl",
    "xls_dslx_ir_impl",
    "xls_ir_opt_ir_attrs",
    "xls_ir_opt_ir_impl",
)
load(
    "//xls/build_rules:xls_ir_wrapper_rules.bzl",
    "FUNCTION_WRAPPER_TYPE",
    "FUZZTEST_WRAPPER_TYPE",
    "generate_jit_wrapper",
)
load(
    "//xls/build_rules:xls_providers.bzl",
    "DslxInfo",
)
load(
    "//xls/build_rules:xls_toolchains.bzl",
    "xls_toolchain_attrs",
)

_DSLX_FUZZ_TEST_DEPS = [
    "@googletest//:gtest",
    "//xls/common/fuzzing:fuzztest",
    "@abseil-cpp//absl/status",
    "@abseil-cpp//absl/status:statusor",
    "//xls/common:xls_gunit_main",
    "//xls/common/status:matchers",
    "//xls/common/status:status_macros",
    "//xls/interpreter:evaluator_options",
    "//xls/ir",
    "//xls/ir:bits",
    "//xls/ir:type",
    "//xls/ir:value",
    "//xls/ir:value_test_util",
    "//xls/ir:xls_type_cc_proto",
    "//xls/jit:function_base_jit_wrapper",
    "//xls/public:value",
]

def _dslx_fuzz_test_impl(ctx):
    """Implementation of the 'dslx_fuzz_test' rule."""
    name = ctx.attr.name
    test_function = ctx.attr.test_function
    jit_namespace = ctx.attr.jit_namespace

    unopt_ir_file = ctx.actions.declare_file(name + "_" + test_function + ".ir")
    ir_file_info, _dslx_info, ir_conv_info, _built_files, ir_runfiles = xls_dslx_ir_impl(
        ctx,
        dslx_top = test_function,
        ir_file = unopt_ir_file,
        ir_conv_args = {},
        library = ctx.attr.library,
    )

    if ctx.attr.opt_ir:
        ir_file_info, _opt_ir_arg_info, _opt_built_files, opt_runfiles = xls_ir_opt_ir_impl(
            ctx,
            ir_file_info,
            ir_conv_info.original_input_files,
            basename = name + "_" + test_function,
            opt_ir_args = {},
            top = "",
            debug_srcs = [],
        )
        ir_runfiles = ir_runfiles.merge(opt_runfiles)

    ir_file = ir_file_info.ir_file

    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    with_msan = cc_common.is_enabled(
        feature_configuration = feature_configuration,
        feature_name = "msan",
    )

    if ctx.target_platform_has_constraint(
        ctx.attr._aarch64_constraint[platform_common.ConstraintValueInfo],
    ):
        aot_target = "aarch64"
    elif ctx.target_platform_has_constraint(
        ctx.attr._x86_64_constraint[platform_common.ConstraintValueInfo],
    ):
        aot_target = "x86_64"
    else:
        aot_target = "native"

    aot_info, _aot_default_info, aot_cc_info = xls_aot_compile(
        ctx = ctx,
        ir_file = ir_file,
        top = "",
        top_type = FUNCTION_WRAPPER_TYPE,
        with_msan = with_msan,
        llvm_opt_level = 3,
        aot_target = aot_target,
        enable_llvm_coverage = True,
    )

    lib_name = name + "_lib"
    lib_class_name = name + "_" + test_function + "_lib"
    lib_source_filename = lib_name + ".cc"
    lib_header_filename = lib_name + ".h"

    lib_wrapper_info, _lib_files, lib_runfiles = generate_jit_wrapper(
        ctx = ctx,
        ir_file = ir_file,
        top = "",
        wrapper_type = FUNCTION_WRAPPER_TYPE,
        jit_wrapper_args = {
            "class_name": lib_class_name,
            "namespace": jit_namespace,
        },
        aot_info_file = aot_info.proto_file,
        source_filename = lib_source_filename,
        header_filename = lib_header_filename,
    )

    fuzzer_name = name + "_fuzzer"
    fuzzer_source_filename = fuzzer_name + ".cc"
    if ctx.label.package:
        lib_header_path = "%s/%s" % (ctx.label.package, lib_header_filename)
    else:
        lib_header_path = lib_header_filename

    fuzzer_wrapper_info, _fuzzer_files, fuzzer_runfiles = generate_jit_wrapper(
        ctx = ctx,
        ir_file = ir_file,
        top = "",
        wrapper_type = FUZZTEST_WRAPPER_TYPE,
        jit_wrapper_args = {
            "class_name": name + "_" + test_function + "_Fuzzer",
            "namespace": jit_namespace,
            "lib_class_name": jit_namespace + "::" + lib_class_name,
            "lib_header_path": lib_header_path,
        },
        aot_info_file = aot_info.proto_file,
        source_filename = fuzzer_source_filename,
        header_filename = None,
    )

    malloc_dep = (
        ctx.attr._default_malloc if ctx.fragments.cpp.custom_malloc != None else ctx.attr._malloc
    )
    dep_targets = ctx.attr._deps + [ctx.attr._libc_runtime, malloc_dep]
    compilation_contexts = [dep[CcInfo].compilation_context for dep in dep_targets]
    linking_contexts = [aot_cc_info.linking_context] + [
        dep[CcInfo].linking_context
        for dep in dep_targets
    ]

    (_compilation_context, compilation_outputs) = cc_common.compile(
        name = name,
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        srcs = [
            lib_wrapper_info.source_file,
            fuzzer_wrapper_info.source_file,
        ],
        private_hdrs = [lib_wrapper_info.header_file],
        compilation_contexts = compilation_contexts,
    )

    linking_outputs = cc_common.link(
        name = name,
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        compilation_outputs = compilation_outputs,
        linking_contexts = linking_contexts,
        output_type = "executable",
        link_deps_statically = True,
    )

    runfiles = ctx.runfiles().merge_all(
        [ir_runfiles, lib_runfiles, fuzzer_runfiles] + [
            dep[DefaultInfo].default_runfiles
            for dep in dep_targets
        ],
    )

    return [
        DefaultInfo(
            executable = linking_outputs.executable,
            files = depset([linking_outputs.executable]),
            runfiles = runfiles,
        ),
    ]

dslx_fuzz_test = rule(
    doc = """A build rule that generates and runs a C++ FuzzTest for a DSLX function.

The rule converts the DSLX library to XLS IR (and optionally optimizes it),
AOT-compiles the IR with LLVM coverage instrumentation, generates C++ JIT and
FuzzTest wrappers, and compiles and links the test executable.
""",
    implementation = _dslx_fuzz_test_impl,
    attrs = dicts.add(
        {
            "library": attr.label(
                doc = "The xls_dslx_library target containing the DSLX code.",
                mandatory = True,
                providers = [DslxInfo],
            ),
            "test_function": attr.string(
                doc = "The name of the top-level DSLX function to be fuzzed.",
                mandatory = True,
            ),
            "opt_ir": attr.bool(
                doc = "Whether to generate optimized IR or not.",
                default = False,
            ),
            "jit_namespace": attr.string(
                doc = "The C++ namespace for the generated JIT wrapper class.",
                default = "dslx_fuzztest::impl",
            ),
            "_deps": attr.label_list(
                default = _DSLX_FUZZ_TEST_DEPS,
                providers = [CcInfo],
            ),
            "_libc_runtime": attr.label(
                default = Label("//xls/dev_tools/embed_data:compat_libc_dep"),
                providers = [CcInfo],
            ),
            "_malloc": attr.label(
                default = Label("@bazel_tools//tools/cpp:malloc"),
                providers = [CcInfo],
            ),
            "_default_malloc": attr.label(
                default = configuration_field(fragment = "cpp", name = "custom_malloc"),
            ),
            "_extra_ir_convert_flags": attr.label(
                default = Label("//xls/common/config:extra_ir_convert_args"),
            ),
            "_aarch64_constraint": attr.label(
                default = Label("@platforms//cpu:aarch64"),
            ),
            "_x86_64_constraint": attr.label(
                default = Label("@platforms//cpu:x86_64"),
            ),
        },
        dicts.pick(xls_ir_opt_ir_attrs, [
            "pass_pipeline",
            "ram_rewrites",
            "_default_pass_pipeline",
            "_extra_opt_flags",
            "_profile_opt_passes",
        ]),
        xls_aot_env_attrs,
        dicts.pick(xls_toolchain_attrs, [
            "_xls_ir_converter_tool",
            "_xls_opt_ir_tool",
            "_xls_aot_compiler_tool",
            "_xls_aot_compiler_segment_tool",
            "_xls_aot_generate_compiler_segments_tool",
            "_xls_jit_wrapper_tool",
        ]),
    ),
    fragments = ["cpp"],
    toolchains = use_cpp_toolchain(),
    test = True,
)
