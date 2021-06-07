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

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "get_mangled_ir_symbol",
    "xls_benchmark_ir",
    "xls_dslx_opt_ir",
    "xls_eval_ir_test",
)

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],
)

xls_dslx_opt_ir(
    name = "idct_chen",
    src = "idct_chen.x",
    ir_file = "idct_chen.ir",
    opt_ir_args = {
        "entry": get_mangled_ir_symbol("idct_chen", "idct"),
    },
    # Note: times out in fastbuild 2020-11-09.
    tags = ["optonly"],
)

# TODO: 2021-05-28 Add xls_ir_equivalence_test. Note: times out.
xls_eval_ir_test(
    name = "idct_chen_eval_ir_test",
    src = "idct_chen.ir",
    ir_eval_args = {
        "entry": get_mangled_ir_symbol("idct_chen", "idct"),
    },
    # Note: times out in fastbuild 2020-11-09.
    tags = ["optonly"],
)

xls_benchmark_ir(
    name = "idct_chen_benchmark_ir",
    src = "idct_chen.ir",
    benchmark_ir_args = {
        "entry": get_mangled_ir_symbol("idct_chen", "idct"),
    },
    # Note: times out in fastbuild 2020-11-09.
    tags = ["optonly"],
)
