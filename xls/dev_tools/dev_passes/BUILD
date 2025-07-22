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

load(
    "//xls/build_rules:xls_pass_rules.bzl",
    "xls_pass",
    "xls_pass_registry",
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
        "//xls/passes",
        ":remove_assert_and_cover_pass",
        ":proc_state_legalization_pass_shim",
        ":literalize_zero_bits_pass",
    ],
    tags = ["keep_dep"],
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
        "@com_google_absl//absl/status:statusor",
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
        "@com_google_absl//absl/status:statusor",
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
        "@com_google_absl//absl/status:statusor",
    ],
)
