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

"""Initializes llvm-bazel project for building llvm."""

load("@llvm-bazel//utils/bazel:configure.bzl", "llvm_configure", "llvm_disable_optional_support_deps")

def initialize():
    # This macro creates a new repo containing the LLVM source code downloaded
    # as llvm-project-raw and the BUILD files from llvm-bazel (contained in the
    # directory llvm-project-overlay). This new repo is constructed as symlinks
    # into these two directories.
    llvm_configure(
        # Name of resulting repo (e.g., "@llvm-project")
        name = "llvm-project",
        targets = ["AArch64", "X86"],
    )
    llvm_disable_optional_support_deps()
