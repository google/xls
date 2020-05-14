# Copyright 2020 Google LLC
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

"""Loads the Yosys framework for Verilog synthesis."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        http_archive,
        name = "at_clifford_yosys",
        urls = [
            "https://github.com/YosysHQ/yosys/archive/aafaeb66dfd839b8223059884d2741dadc9e2d92.zip",
        ],
        strip_prefix = "yosys-aafaeb66dfd839b8223059884d2741dadc9e2d92",
        sha256 = "d8156497c882fb6754597b6193a65966b83cd81e362174de5d986b6a5fce05c3",
        build_file = Label("//dependency_support:at_clifford_yosys/bundled.BUILD.bazel"),
    )
