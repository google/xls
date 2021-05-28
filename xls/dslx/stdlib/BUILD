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

load("//xls/build_rules:xls_build_defs.bzl", "xls_dslx_test")

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_test(
    name = "std",
    src = "std.x",
)

xls_dslx_test(
    name = "acm_random",
    src = "acm_random.x",
)

xls_dslx_test(
    name = "apfloat",
    src = "apfloat.x",
)

xls_dslx_test(
    name = "bfloat16",
    src = "bfloat16.x",
)

xls_dslx_test(
    name = "float32",
    src = "float32.x",
)

xls_dslx_test(
    name = "float64",
    src = "float64.x",
)

filegroup(
    name = "x_files",
    srcs = glob(["*.x"]),
)
