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
    LLVM_COMMIT = "628dda08b82fcedcd9e89c9ef7850388e988bf68"  # 2021-04-07
    LLVM_SHA256 = "b48c8c63c17631cc0160e1359c1e977188aa6cf5924cfd4b5664397effe65f30"
    LLVM_BAZEL_SHA256 = "259684aa19af62201bd6a261faf315aff8a17dc8bcafcaed372b14108a872f71"

    maybe(
        http_archive,
        name = "llvm-bazel",
        sha256 = LLVM_BAZEL_SHA256,
        strip_prefix = "llvm-bazel-llvm-project-{llvm_commit}/llvm-bazel".format(llvm_commit = LLVM_COMMIT),
        url = "https://github.com/google/llvm-bazel/archive/refs/tags/llvm-project-{llvm_commit}.tar.gz".format(llvm_commit = LLVM_COMMIT),
    )

    maybe(
        http_archive,
        name = "llvm-project-raw",
        build_file_content = "#empty",
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-" + LLVM_COMMIT,
        url = "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
    )
