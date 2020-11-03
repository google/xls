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
    name = "network_proto",
    srcs = ["network_config.proto"],
)

cc_proto_library(
    name = "network_cc_proto",
    deps = [
        ":network_proto",
    ],
)

proto_library(
    name = "network_builder_arguments_proto",
    srcs = ["network_config_builder_arguments.proto"],
)

cc_proto_library(
    name = "network_builder_arguments_cc_proto",
    deps = [
        ":network_builder_arguments_proto",
    ],
)

cc_library(
    name = "network",
    hdrs = ["network_config_builder.h"],
    deps = [
        ":network_builder_arguments_cc_proto",
        ":network_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:status_macros",
    ],
)

cc_library(
    name = "network_factory",
    srcs = [
        "network_config_builder_factory.cc",
    ],
    hdrs = ["network_config_builder_factory.h"],
    deps = [
        ":network",
        ":network_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:module_initializer",
    ],
)
