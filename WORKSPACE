# Copyright 2022 The XLS Authors
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

workspace(name = "com_google_xls")

# Load and configure a hermetic LLVM based C/C++ toolchain. This is done here
# and not in load_external.bzl because it requires several sequential steps of
# declaring archives and using things in them, which is awkward to do in .bzl
# files because it's not allowed to use `load` inside of a function.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Released 2023-09-20, current as of 2024-06-26 (but there is already a 0.0.10rc1)
# Needs to be loaded first, as llvm toolchain has an ancient version of this.
http_archive(
    name = "rules_cc",
    urls = ["https://github.com/bazelbuild/rules_cc/releases/download/0.0.9/rules_cc-0.0.9.tar.gz"],
    sha256 = "2037875b9a4456dce4a79d112a8ae885bbc4aad968e6587dca6e64f3a0900cdf",
    strip_prefix = "rules_cc-0.0.9",
)

# Commit on  2024-07-19, current as of 2024-07-21.
http_archive(
    name = "toolchains_llvm",
    integrity = "sha256-RVp0bZsDrelQAtxswWOLB4j8zCXQy/cSe1m3wL/E2PU=",
    strip_prefix = "toolchains_llvm-01132cfdae7d7187a885cf79d5a3ac1ed8a02e5a",
    url = "https://github.com/bazel-contrib/toolchains_llvm/archive/01132cfdae7d7187a885cf79d5a3ac1ed8a02e5a.tar.gz",
)

load("@toolchains_llvm//toolchain:deps.bzl", "bazel_toolchain_dependencies")

bazel_toolchain_dependencies()

load("@toolchains_llvm//toolchain:rules.bzl", "llvm_toolchain")

llvm_toolchain(
    name = "llvm_toolchain",
    llvm_version = "17.0.6",
)

load("@llvm_toolchain//:toolchains.bzl", "llvm_register_toolchains")

llvm_register_toolchains()

load("//dependency_support:load_external.bzl", "load_external_repositories")

load_external_repositories()

load(
  "@rules_python//python:repositories.bzl",
  "py_repositories",
  "python_register_toolchains",
)

# Must be called before using anything from rules_python.
# https://github.com/bazelbuild/rules_python/issues/1560#issuecomment-1815118394
py_repositories()

python_register_toolchains(
    name = "project_python",
    python_version = "3.11",

    # Required for our containerized CI environments; we do not recommend
    # building XLS as root normally.
    ignore_root_user_error = True,
)

# gRPC deps should be loaded before initializing other repos. Otherwise, various
# errors occur during repo loading and initialization.
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load("//dependency_support:initialize_external.bzl", "initialize_external_repositories")

initialize_external_repositories()

load("@xls_pip_deps//:requirements.bzl", xls_pip_install_deps = "install_deps")

xls_pip_install_deps()

# Loading the extra deps must be called after initialize_eternal_repositories or
# the call to pip_parse fails.
load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

grpc_extra_deps()
