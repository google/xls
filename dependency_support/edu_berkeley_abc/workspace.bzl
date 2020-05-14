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

"""Loads the ABC system for sequential synthesis and verification, used by yosys."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        http_archive,
        name = "edu_berkeley_abc",
        urls = [
            "https://github.com/berkeley-abc/abc/archive/a918e2dab1f951eb7e869f07b57f648b9a583561.zip",
        ],
        strip_prefix = "abc-a918e2dab1f951eb7e869f07b57f648b9a583561",
        sha256 = "e2cb19f5c6a41cd059d749beb066afdc7759a2c6da822a975a73cfcd014ea3e6",
        build_file = Label("//dependency_support:edu_berkeley_abc/bundled.BUILD.bazel"),
    )
