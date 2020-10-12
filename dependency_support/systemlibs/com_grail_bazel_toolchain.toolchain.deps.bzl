# Copyright 2020 The XLS Authors
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

load("@//dependency_support:repo.bzl", "xls_http_archive")

def bazel_toolchain_dependencies():

    xls_http_archive(
        name = "llvm_toolchain",
        urls = [ "https://github.com/grailbio/bazel-toolchain/archive/f4c17a3ae40f927ff62cc0fb8fe22b1530871807.zip", ],
        sha256 = "715fd98d566ed1304cb53e0c640427cf0916ec6db89588e3ac2b6a87632276d4",
        system_build_file = "@//dependency_support/systemlibs:llvm_toolchain.BUILD",
        system_link_files = {
            "@//dependency_support/systemlibs:llvm_toolchain.toolchains.bzl": "toolchains.bzl",
            "@//dependency_support/systemlibs:llvm_toolchain.cc_toolchain_config.bzl": "cc_toolchain_config.bzl",
        },
    )

    if not native.existing_rule("rules_cc"):
        xls_http_archive(
            name = "rules_cc",
            sha256 = "b6f34b3261ec02f85dbc5a8bdc9414ce548e1f5f67e000d7069571799cb88b25",
            strip_prefix = "rules_cc-726dd8157557f1456b3656e26ab21a1646653405",
            urls = ["https://github.com/bazelbuild/rules_cc/archive/726dd8157557f1456b3656e26ab21a1646653405.tar.gz"],
        )
