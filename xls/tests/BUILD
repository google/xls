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

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_test(
    name = "basic_ir_ops_test",
    srcs = ["basic_ir_ops_test.cc"],
    shard_count = 2,
    deps = [
        "@com_google_absl//absl/strings",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "ir_types_test",
    srcs = ["ir_types_test.cc"],
    deps = [
        "//xls/ir",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "dslx_optimization_test",
    srcs = ["dslx_optimization_test.cc"],
    data = ["//xls/dslx:ir_converter_main"],
    deps = [
        "//xls/common:subprocess",
        "//xls/common/file:get_runfile_path",
        "//xls/common/file:temp_file",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:ir_test_base",
        "//xls/ir:op",
        "//xls/passes:standard_pipeline",
        "@com_google_googletest//:gtest_main",
    ],
)
