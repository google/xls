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

load("//xls/build:build_defs.bzl", "dslx_test")

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],
)

dslx_test(
    name = "idct_chen",
    srcs = ["idct_chen.x"],
    entry = "__idct_chen__idct",
    # Note: times out.
    prove_unopt_eq_opt = False,
    # Note: times out in fastbuild 2020-11-09.
    tags = ["optonly"],
)
