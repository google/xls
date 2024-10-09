# Copyright 2024 The XLS Authors
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

load("@xls_pip_deps//:requirements.bzl", "requirement")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_users"],
    licenses = ["notice"],
)

py_library(
    name = "channel",
    srcs = ["channel.py"],
    deps = [
        ":xlsstruct",
        requirement("cocotb"),
        requirement("cocotb_bus"),
    ],
)

py_library(
    name = "memory",
    srcs = ["memory.py"],
    deps = [
        requirement("cocotbext-axi"),
    ],
)

py_library(
    name = "scoreboard",
    srcs = ["scoreboard.py"],
    deps = [
        ":channel",
        ":xlsstruct",
        requirement("cocotb"),
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        requirement("cocotb"),
        "//xls/common:runfiles",
    ],
)

py_library(
    name = "xlsstruct",
    srcs = ["xlsstruct.py"],
    deps = [
        requirement("cocotb"),
    ],
)

py_library(
    name = "data_generator",
    srcs = ["data_generator.py"],
    deps = [
        "//xls/common:runfiles",
        "@zstd//:decodecorpus",
    ],
)
