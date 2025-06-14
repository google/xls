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
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_fmt_test",
    "xls_dslx_library",
    "xls_dslx_prove_quickcheck_test",
    "xls_dslx_test",
)

xls_dslx_library(
    name = "common",
    srcs = ["common.x"],
)

xls_dslx_library(
    name = "sign_magnitude",
    srcs = ["sign_magnitude.x"],
    deps = [":common"],
)

xls_dslx_prove_quickcheck_test(
    name = "sign_magnitude_add_sub_test",
    library = ":sign_magnitude",
)

xls_dslx_library(
    name = "far_path",
    srcs = ["far_path.x"],
    deps = [
        ":common",
        ":sign_magnitude",
    ],
)

xls_dslx_test(
    name = "far_path_test",
    library = ":far_path",
)

xls_dslx_library(
    name = "near_path",
    srcs = ["near_path.x"],
    deps = [
        ":common",
        ":sign_magnitude",
    ],
)

xls_dslx_test(
    name = "near_path_test",
    library = ":near_path",
)

xls_dslx_library(
    name = "dual_path",
    srcs = ["dual_path.x"],
    deps = [
        ":common",
        ":far_path",
        ":near_path",
    ],
)

xls_dslx_test(
    name = "dual_path_test",
    library = ":dual_path",
)

# -- we only want to prove quickcheck for the bf16 target so we break it out here

xls_dslx_library(
    name = "quickcheck_add_dual_path_bf16",
    srcs = ["quickcheck_add_dual_path_bf16.x"],
    deps = [":dual_path"],
)

xls_dslx_test(
    name = "quickcheck_add_dual_path_bf16_test",
    library = ":quickcheck_add_dual_path_bf16",
)

# This proves in a reasonable amount of time, ~10 seconds on my desktop as of 2025-02-24.
xls_dslx_prove_quickcheck_test(
    name = "quickcheck_add_dual_path_bf16_proof_test",
    library = ":quickcheck_add_dual_path_bf16",
)

xls_dslx_library(
    name = "quickcheck_add_dual_path_f32",
    srcs = ["quickcheck_add_dual_path_f32.x"],
    deps = [":dual_path"],
)

xls_dslx_test(
    name = "quickcheck_add_dual_path_f32_test",
    library = ":quickcheck_add_dual_path_f32",
)

[xls_dslx_fmt_test(
    name = file.replace(".x", "_fmt_test"),
    src = file,
) for file in glob(["*.x"])]
