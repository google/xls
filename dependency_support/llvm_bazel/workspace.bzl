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

"""Loads llvm-bazel and llvm projects."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    """Initialize the llvm-bazel (and llvm-project) repository."""
    LLVM_COMMIT = "6a7a2ee8161da84d9a58a88b497b0b47c8df99f3"
    LLVM_SHA256 = "364445a975b91328c99cc61d76b4d891e1846131551aad5d7e3a4e4a0c4a2d91"

    maybe(
        http_archive,
        name = "llvm-bazel",
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-{llvm_commit}/utils/bazel".format(llvm_commit = LLVM_COMMIT),
        url = "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
    )

    maybe(
        http_archive,
        name = "llvm-project-raw",
        build_file_content = "#empty",
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-" + LLVM_COMMIT,
        url = "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
    )
