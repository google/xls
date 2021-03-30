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
    LLVM_COMMIT = "22405685794a4908ae64e71d97532f8ab6d34f5c"  # 2021-03-29
    LLVM_BAZEL_COMMIT = "3ae351146899f06d2c763e3dbbe3387f9e7c0408"
    LLVM_BAZEL_SHA256 = "ea1fe96c57d425cd0d7eda2774dce55242c6292ba67983923919fecdff676d1a"
    LLVM_SHA256 = "d08dc70b92e290a9ff33c2de294ce50065de322ef3ab427e2617321eb4eb94ec"

    maybe(
        http_archive,
        name = "llvm-bazel",
        sha256 = LLVM_BAZEL_SHA256,
        strip_prefix = "llvm-bazel-{commit}/llvm-bazel".format(commit = LLVM_BAZEL_COMMIT),
        url = "https://github.com/google/llvm-bazel/archive/{commit}.tar.gz".format(commit = LLVM_BAZEL_COMMIT),
    )

    maybe(
        http_archive,
        name = "llvm-project-raw",
        build_file_content = "#empty",
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-" + LLVM_COMMIT,
        url = "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
    )
