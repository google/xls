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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//xls/build_rules:xls_build_defs.bzl", "xls_diff_test")
load(
    "//xls/build_rules:xls_pass_rules.bzl",
    "xls_pass",
    "xls_pass_registry",
)
load(
    "//xls/passes/tools:generate_documentation_rule.bzl",
    "xls_generate_documentation",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

# Passes useful for dev tools but not in other circumstances.

# Additional passes which can be useful for making dev tools but are generally
# not helpful for actual optimization.
xls_pass_registry(
    name = "dev_passes",
    passes = [
        ":remove_assert_and_cover_pass",
        ":proc_state_legalization_pass_shim",
        ":literalize_zero_bits_pass",
        ":to_zero_ext_pass",
        ":decompose_dataflow_pass",
        ":remove_one_hot_select_pass",
        ":remove_identifiers_pass",
    ],
    tags = ["keep_dep"],
)

# This generates the passes_list.md file.
#
# TODO(https://github.com/google/xls/issues/2556) This has complicated
# requirements on libcxx being from the same version of llvm as the clang
# depenedencies. Internally at google this works automatically but in OSS this
# doesn't work.
xls_generate_documentation(
    name = "rebuild_documentation",
    codelink_format = "http://github.com/google/xls/tree/main%s",
    # Specifically generate for the dev passes regardless of what the default passes are.
    passes = ":dev_passes",
    # Requirements on libcxx being explicit in include path.
    tags = [
        "manual",
        "noasan",
        "nomsan",
    ],
    template = "dev_passes.md.tmpl",
)

# TODO(allight): Making this runnable with asan/msan would be nice but not particularly important.
xls_diff_test(
    name = "dev_passes_md_up_to_date_test",
    failure_message = """Documentation dev_passes_list.md is out of date.

Please update it using dev_tools/rebuild_golden_files.sh.""",
    file = ":rebuild_documentation",
    golden = "//docs_src:dev_passes_list.md",
    # Requirements on libcxx being explicit in include path.
    tags = [
        "manual",
        "no_update_golden",
        "noasan",
        "nomsan",
    ],
)

xls_pass(
    name = "remove_assert_and_cover_pass",
    srcs = ["remove_assert_and_cover_pass.cc"],
    hdrs = ["remove_assert_and_cover_pass.h"],
    pass_class = "AssertAndCoverRemovalPass",
    deps = [
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/passes:optimization_pass",
        "//xls/passes:pass_base",
        "@abseil-cpp//absl/status:statusor",
    ],
)

xls_pass(
    name = "proc_state_legalization_pass_shim",
    srcs = ["proc_state_legalization_pass_shim.cc"],
    hdrs = ["proc_state_legalization_pass_shim.h"],
    pass_class = "ProcStateLegalizationPassShim",
    deps = [
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/passes:optimization_pass",
        "//xls/passes:pass_base",
        "//xls/scheduling:proc_state_legalization_pass",
        "//xls/scheduling:scheduling_pass",
        "@abseil-cpp//absl/status:statusor",
    ],
)

xls_pass(
    name = "literalize_zero_bits_pass",
    srcs = ["literalize_zero_bits_pass.cc"],
    hdrs = ["literalize_zero_bits_pass.h"],
    pass_class = "LiteralizeZeroBits",
    deps = [
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:value",
        "//xls/passes:optimization_pass",
        "//xls/passes:pass_base",
        "@abseil-cpp//absl/status:statusor",
    ],
)

xls_pass(
    name = "to_zero_ext_pass",
    srcs = ["to_zero_ext_pass.cc"],
    hdrs = ["to_zero_ext_pass.h"],
    pass_class = "ToZeroExtPass",
    deps = [
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:node_util",
        "//xls/ir:op",
        "//xls/passes:optimization_pass",
        "//xls/passes:partial_info_query_engine",
        "//xls/passes:pass_base",
        "@abseil-cpp//absl/status:statusor",
    ],
)

cc_test(
    name = "to_zero_ext_pass_test",
    srcs = ["to_zero_ext_pass_test.cc"],
    deps = [
        ":to_zero_ext_pass",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/passes:optimization_pass",
        "//xls/passes:pass_base",
        "@abseil-cpp//absl/status:statusor",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "pass_overrides",
    srcs = ["pass_overrides.cc"],
    hdrs = ["pass_overrides.h"],
    deps = [
        "//xls/common/status:status_macros",
        "//xls/passes:optimization_pass",
        "//xls/passes:optimization_pass_registry",
        "//xls/tools:opt",
        "@abseil-cpp//absl/status:statusor",
    ],
)

cc_test(
    name = "pass_overrides_test",
    srcs = ["pass_overrides_test.cc"],
    deps = [
        ":pass_overrides",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
        "//xls/passes:optimization_pass",
        "//xls/passes:optimization_pass_pipeline",
        "//xls/passes:optimization_pass_registry",
        "//xls/passes:pass_base",
        "//xls/tools:opt",
        "@abseil-cpp//absl/status:statusor",
        "@googletest//:gtest",
    ],
)

xls_pass(
    name = "decompose_dataflow_pass",
    srcs = ["decompose_dataflow_pass.cc"],
    hdrs = ["decompose_dataflow_pass.h"],
    pass_class = "DecomposeDataflowPass",
    deps = [
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/data_structures:leaf_type_tree",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:node_util",
        "//xls/ir:op",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/passes:dataflow_visitor",
        "//xls/passes:optimization_pass",
        "//xls/passes:pass_base",
        "//xls/passes:query_engine",
        "//xls/passes:stateless_query_engine",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@cppitertools",
    ],
)

cc_test(
    name = "decompose_dataflow_pass_test",
    srcs = ["decompose_dataflow_pass_test.cc"],
    deps = [
        ":decompose_dataflow_pass",
        "//xls/common:xls_gunit_main",
        "//xls/common/fuzzing:fuzztest",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/fuzzer/ir_fuzzer:ir_fuzz_domain",
        "//xls/fuzzer/ir_fuzzer:ir_fuzz_test_library",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:channel_ops",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
        "//xls/passes:dce_pass",
        "//xls/passes:optimization_pass",
        "//xls/passes:pass_base",
        "//xls/passes:pass_test_helpers",
        "//xls/solvers:ir_equivalence_testutils",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@googletest//:gtest",
    ],
)

xls_pass(
    name = "remove_one_hot_select_pass",
    srcs = ["remove_one_hot_select_pass.cc"],
    hdrs = ["remove_one_hot_select_pass.h"],
    pass_class = "RemoveOneHotSelectPass",
    deps = [
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:node_util",
        "//xls/ir:value_utils",
        "//xls/passes:optimization_pass",
        "//xls/passes:pass_base",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@cppitertools",
    ],
)

cc_test(
    name = "remove_one_hot_select_pass_test",
    srcs = ["remove_one_hot_select_pass_test.cc"],
    deps = [
        ":remove_one_hot_select_pass",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/passes:optimization_pass",
        "//xls/passes:pass_base",
        "//xls/solvers:ir_equivalence_testutils",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@googletest//:gtest",
    ],
)

xls_pass(
    name = "remove_identifiers_pass",
    srcs = ["remove_identifiers_pass.cc"],
    hdrs = ["remove_identifiers_pass.h"],
    pass_class = "RemoveIdentifiersPass",
    deps = [
        "//xls/common/status:status_macros",
        "//xls/dev_tools:remove_identifiers",
        "//xls/ir",
        "//xls/passes:optimization_pass",
        "//xls/passes:pass_base",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status:statusor",
    ],
)

cc_test(
    name = "remove_identifiers_pass_test",
    srcs = ["remove_identifiers_pass_test.cc"],
    deps = [
        ":remove_identifiers_pass",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
        "//xls/passes:optimization_pass",
        "//xls/passes:pass_base",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@googletest//:gtest",
    ],
)
