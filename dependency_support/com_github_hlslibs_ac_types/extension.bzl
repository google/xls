# Copyright 2026 The XLS Authors
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

"""Module extension for ac_types."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _ac_types_extension_impl(
        module_ctx):  # @unused
    # Tagged 2024-02-16 (note: release is lagging tag), current as of 2024-06-26
    http_archive(
        name = "ac_datatypes",
        urls = ["https://github.com/hlslibs/ac_types/archive/refs/tags/4.8.0.tar.gz"],
        sha256 = "238197203f8c6254a1d6ac6884e89e6f4c060bffb7473d336df4a1fb53ba7fab",
        strip_prefix = "ac_types-4.8.0",
        build_file = Label("//dependency_support/com_github_hlslibs_ac_types:bundled.BUILD.bazel"),
    )

ac_types_extension = module_extension(
    implementation = _ac_types_extension_impl,
)
