# Copyright 2020 The XLS Authors
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

# Build rules for DSLX modules.
#
load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_users"],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_library(
    name = "apfloat_fmac_dslx",
    srcs = ["apfloat_fmac.x"],
)

xls_dslx_library(
    name = "fp32_fmac_dslx",
    srcs = ["fp32_fmac.x"],
    deps = [":apfloat_fmac_dslx"],
)

xls_dslx_opt_ir(
    name = "fp32_fmac",
    dslx_top = "fp32_fmac",
    library = ":fp32_fmac_dslx",
)

xls_dslx_test(
    name = "fp32_fmac_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = ":fp32_fmac_dslx",
)
