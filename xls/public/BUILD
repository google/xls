# Copyright 2021 The XLS Authors
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

# Public (exported) XLS APIs.

package(
    default_visibility = ["//xls:xls_users"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "runtime_build_actions",
    srcs = ["runtime_build_actions.cc"],
    hdrs = ["runtime_build_actions.h"],
    data = [
        "//xls/dslx:ir_converter_main",
        "//xls/tools:opt_main",
    ],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/dslx:import_data",
        "//xls/dslx:ir_converter",
        "//xls/dslx:mangle",
        "//xls/dslx:parse_and_typecheck",
        "//xls/passes",
        "//xls/tools:opt",
        "//xls/tools:proto_to_dslx",
    ],
)

cc_test(
    name = "runtime_build_actions_test",
    srcs = ["runtime_build_actions_test.cc"],
    deps = [
        ":runtime_build_actions",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "function_builder",
    hdrs = ["function_builder.h"],
    deps = [
        "//xls/ir:function_builder",
    ],
)

cc_library(
    name = "value",
    hdrs = ["value.h"],
    deps = [
        "//xls/ir:bits",
        "//xls/ir:value",
        "//xls/ir:value_view",
    ],
)
