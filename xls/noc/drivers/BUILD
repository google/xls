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

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "experiment",
    srcs = ["experiment.cc"],
    hdrs = ["experiment.h"],
    deps = [
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/noc/config:network_config_cc_proto",
        "//xls/noc/simulation:common",
        "//xls/noc/simulation:flit",
        "//xls/noc/simulation:global_routing_table",
        "//xls/noc/simulation:network_graph",
        "//xls/noc/simulation:network_graph_builder",
        "//xls/noc/simulation:noc_traffic_injector",
        "//xls/noc/simulation:parameters",
        "//xls/noc/simulation:sim_objects",
        "//xls/noc/simulation:simulator_to_link_monitor_shim",
        "//xls/noc/simulation:simulator_to_traffic_injector_shim",
        "//xls/noc/simulation:traffic_description",
    ],
)

cc_library(
    name = "experiment_factory",
    hdrs = ["experiment_factory.h"],
    deps = [
        ":experiment",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:casts",
    ],
)

cc_test(
    name = "experiment_and_factory_test",
    srcs = ["experiment_and_factory_test.cc"],
    deps = [
        ":experiment",
        ":experiment_factory",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/noc/config:network_config_cc_proto",
        "//xls/noc/config:network_config_proto_builder",
        "//xls/noc/simulation:common",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "sample_experiments",
    srcs = ["sample_experiments.cc"],
    hdrs = ["sample_experiments.h"],
    deps = [
        ":experiment_factory",
        "@com_google_absl//absl/status",
        "//xls/noc/config:network_config_cc_proto",
        "//xls/noc/drivers/samples:aggregate_tree_experiment",
        "//xls/noc/drivers/samples:simple_vc_experiment",
    ],
)

cc_test(
    name = "sample_experiments_test",
    srcs = ["sample_experiments_test.cc"],
    deps = [
        ":experiment_factory",
        ":sample_experiments",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)
