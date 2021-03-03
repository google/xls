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

load("@llvm-bazel//:terminfo.bzl", "llvm_terminfo_disable")
load("@llvm-bazel//:zlib.bzl", "llvm_zlib_disable")
load("@llvm-bazel//:configure.bzl", "llvm_configure")

def initialize():
    llvm_terminfo_disable(
        name = "llvm_terminfo",
    )
    llvm_zlib_disable(
        name = "llvm_zlib",
    )

    # This macro creates a new repo containing the LLVM source code downloaded
    # as llvm-project-raw and the BUILD files from llvm-bazel (contained in the
    # directory llvm-project-overlay). This new repo is constructed as symlinks
    # into these two directories.
    llvm_configure(
        # Name of resulting repo (e.g., "@llvm-project")
        name = "llvm-project",

        # The path to the directory containing the LLVM BUILD files within
        # llvm-bazel.
        overlay_path = "llvm-project-overlay",
        overlay_workspace = "@llvm-bazel//:WORKSPACE",

        # Project containing the llvm source code downloaded with http_archive.
        src_path = ".",
        src_workspace = "@llvm-project-raw//:WORKSPACE",
    )
