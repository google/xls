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

# cc_proto_library is used in this file

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

proto_library(
    name = "topology_options_network_config_builder_proto",
    srcs = ["topology_options_network_config_builder.proto"],
)

cc_proto_library(
    name = "topology_options_network_config_builder_cc_proto",
    deps = [
        ":topology_options_network_config_builder_proto",
    ],
)

proto_library(
    name = "component_configuration_options_network_config_builder_proto",
    srcs = ["component_configuration_options_network_config_builder.proto"],
    deps = ["//xls/noc/config:network_config_proto"],
)

cc_proto_library(
    name = "component_configuration_options_network_config_builder_cc_proto",
    deps = [
        ":component_configuration_options_network_config_builder_proto",
    ],
)

proto_library(
    name = "routing_algorithm_options_network_config_builder_proto",
    srcs = ["routing_algorithm_options_network_config_builder.proto"],
)

cc_proto_library(
    name = "routing_algorithm_options_network_config_builder_cc_proto",
    deps = [
        ":routing_algorithm_options_network_config_builder_proto",
    ],
)

proto_library(
    name = "network_config_builder_options_proto",
    srcs = ["network_config_builder_options.proto"],
    deps = [
        ":component_configuration_options_network_config_builder_proto",
        ":routing_algorithm_options_network_config_builder_proto",
        ":topology_options_network_config_builder_proto",
    ],
)

cc_proto_library(
    name = "network_config_builder_options_cc_proto",
    deps = [
        ":network_config_builder_options_proto",
    ],
)
