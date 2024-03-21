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

# pytype tests are present in this file

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_test(
    name = "basic_ir_ops_test",
    srcs = ["basic_ir_ops_test.cc"],
    shard_count = 2,
    deps = [
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "ir_types_test",
    srcs = ["ir_types_test.cc"],
    deps = [
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/ir",
        "//xls/ir:ir_test_base",
    ],
)

cc_test(
    name = "dslx_optimization_test",
    srcs = ["dslx_optimization_test.cc"],
    data = ["//xls/dslx/ir_convert:ir_converter_main"],
    deps = [
        "//xls/common:subprocess",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:get_runfile_path",
        "//xls/common/file:temp_file",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx:mangle",
        "//xls/ir",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/ir:op",
        "//xls/passes:optimization_pass_pipeline",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

py_test(
    name = "documentation_test",
    srcs = ["documentation_test.py"],
    data = [
        "//docs_src:dslx_reference",
        "//docs_src/tutorials:all_files",
        "//xls/dslx:interpreter_main",
        "//xls/dslx/tests:mod_imported_file",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
    ],
)
