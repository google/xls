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

# Dependencies locked to avoid transitive dependencies on llvm.
# Use the XLS[cc] binary for C++ to XLS IR translation, rather than directly
#  depending on the translation library.
# Email xls-team@ or seanhaskell@ if a dependency is desired.

# cc_proto_library is used in this file
load("//xls/build_rules:py_proto_library.bzl", "xls_py_proto_library")

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

proto_library(
    name = "hls_block_proto",
    srcs = ["hls_block.proto"],
    features = ["-proto_dynamic_mode_static_link"],
)

cc_proto_library(
    name = "hls_block_cc_proto",
    deps = [":hls_block_proto"],
)

xls_py_proto_library(
    name = "hls_block_py_pb2",
    srcs = ["hls_block.proto"],
    internal_deps = [":hls_block_proto"],
)
