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

# Commit on  2024-01-19, current as of 2024-02-01.
http_archive(
    name = "toolchains_llvm",
    #sha256 = "5b01ab0cf15ddf9c7e4412964238e24ca869ba0e2d0825b70d055f2b2cd895a9",
    sha256 = "f84011575162219292bb0b69725129f55f0e338f3282b22ac529e08438247f45",
    strip_prefix = "toolchains_llvm-05f0bc1f4b1b12ad7ce0ad5ef9235a94ff39ff54",
    url = "https://github.com/grailbio/bazel-toolchain/archive/05f0bc1f4b1b12ad7ce0ad5ef9235a94ff39ff54.tar.gz",
)

load("@toolchains_llvm//toolchain:deps.bzl", "bazel_toolchain_dependencies")

bazel_toolchain_dependencies()

load("@toolchains_llvm//toolchain:rules.bzl", "llvm_toolchain")

llvm_toolchain(
    name = "llvm_toolchain",
    llvm_version = "17.0.2",
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

py_repositories()

python_register_toolchains(
    name = "python39",
    python_version = "3.9",

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

load("@rules_hdl_pip_deps//:requirements.bzl", rules_hdl_pip_install_deps = "install_deps")

rules_hdl_pip_install_deps()

load("@xls_pip_deps//:requirements.bzl", xls_pip_install_deps = "install_deps")

xls_pip_install_deps()

# Loading the extra deps must be called after initialize_eternal_repositories or
# the call to pip_parse fails.
load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

grpc_extra_deps()
