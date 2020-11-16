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

# cc_proto_library is used in this file

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

proto_library(
    name = "network_config_proto",
    srcs = ["network_config.proto"],
)

cc_proto_library(
    name = "network_config_cc_proto",
    deps = [
        ":network_config_proto",
    ],
)

proto_library(
    name = "network_config_builder_options_proto",
    srcs = ["network_config_builder_options.proto"],
)

cc_proto_library(
    name = "network_config_builder_options_cc_proto",
    deps = [
        ":network_config_builder_options_proto",
    ],
)

cc_library(
    name = "network_config_builder",
    hdrs = ["network_config_builder.h"],
    deps = [
        ":network_config_builder_options_cc_proto",
        ":network_config_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:status_macros",
    ],
)

cc_library(
    name = "network_config_builder_factory",
    srcs = [
        "network_config_builder_factory.cc",
    ],
    hdrs = ["network_config_builder_factory.h"],
    deps = [
        ":network_config_builder",
        ":network_config_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:module_initializer",
    ],
)

cc_library(
    name = "network_config_builder_util",
    srcs = [
        "network_config_builder_util.cc",
    ],
    hdrs = ["network_config_builder_util.h"],
    deps = [
        ":network_config_builder_options_cc_proto",
        ":network_config_cc_proto",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:integral_types",
        "//xls/common:math_util",
    ],
)

cc_library(
    name = "ring",
    srcs = [
        "ring.cc",
    ],
    deps = [
        ":network_config_builder",
        ":network_config_builder_factory",
        ":network_config_builder_options_cc_proto",
        ":network_config_builder_util",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:math_util",
        "@com_google_protobuf//:protobuf",
    ],
    alwayslink = True,
)
