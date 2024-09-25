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

"""Initializes LLVM."""

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")

def initialize():
    # This macro creates a new repo containing the LLVM source code downloaded
    # as llvm-raw.
    llvm_configure(
        # Name of resulting repo (e.g., "@llvm-project")
        name = "llvm-project",
        # TODO(jpienaar): NVPTX is merely here as a config to generate additional BUILD targets. It
        # is not actually compiled or linked in. Change upstream build deps to enable removing it.
        targets = ["AArch64", "X86", "NVPTX"],
    )
