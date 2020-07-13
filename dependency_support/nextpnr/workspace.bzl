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

"""Loads NextPNR, a portable FPGA place and route tool."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        http_archive,
        name = "nextpnr",
        urls = [
            "https://github.com/YosysHQ/nextpnr/archive/f44498a5301f9f516488fb748c684926be514346.zip",  # 2020-05-27
        ],
        strip_prefix = "nextpnr-f44498a5301f9f516488fb748c684926be514346",
        sha256 = "ee2a3a9f8a3632b28b33f0c7bd64d70e166c7f641184f2b84b606b7d8a67b878",
        build_file = Label("//dependency_support/nextpnr:bundled.BUILD.bazel"),
    )
