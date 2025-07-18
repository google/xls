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
# Load proto_library
# cc_proto_library is used in this file

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_test(
    name = "basic_ir_ops_test",
    srcs = ["basic_ir_ops_test.cc"],
    shard_count = 3,
    deps = [
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
        "//xls/simulation:sim_test_base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "ir_types_test",
    srcs = ["ir_types_test.cc"],
    deps = [
        "//xls/common:xls_gunit_main",
        "//xls/ir:bits",
        "//xls/ir:value",
        "//xls/simulation:sim_test_base",
        "@com_google_absl//absl/types:span",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "dslx_optimization_test",
    srcs = ["dslx_optimization_test.cc"],
    data = ["//xls/dslx/ir_convert:ir_converter_main"],
    deps = [
        "//xls/common:subprocess",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:get_runfile_path",
        "//xls/common/file:temp_file",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx:mangle",
        "//xls/ir",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/ir:op",
        "//xls/passes",
        "//xls/passes:optimization_pass_pipeline",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@googletest//:gtest",
    ],
)

py_test(
    name = "documentation_test",
    srcs = ["documentation_test.py"],
    data = [
        "//docs_src:dslx_documentation_files",
        "//docs_src/tutorials:all_files",
        "//xls/dslx:dslx_fmt",
        "//xls/dslx:interpreter_main",
        "//xls/dslx/tests:mod_imported_file",
    ],
    deps = [
        "//xls/common:runfiles",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
    ],
)

cc_library(
    name = "testbench",
    hdrs = ["testbench.h"],
    deps = [
        ":testbench_thread",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "testbench_builder",
    hdrs = ["testbench_builder.h"],
    deps = [
        ":testbench",
        ":testbench_builder_utils",
    ],
)

cc_library(
    name = "testbench_builder_utils",
    hdrs = ["testbench_builder_utils.h"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "testbench_builder_utils_test",
    srcs = ["testbench_builder_utils_test.cc"],
    deps = [
        ":testbench_builder_utils",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "testbench_thread",
    hdrs = ["testbench_thread.h"],
    deps = [
        "//xls/common:thread",
        "//xls/ir",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
)

proto_library(
    name = "testvector_proto",
    srcs = ["testvector.proto"],
)

cc_proto_library(
    name = "testvector_cc_proto",
    deps = [":testvector_proto"],
)
