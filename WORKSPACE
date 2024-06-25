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

# Commit on  2024-02-13, current as of 2024-02-13.
http_archive(
    name = "toolchains_llvm",
    sha256 = "916ee9a85287058dbd00fdc3659ab92176d9019590f8e8b763cd49d85476e253",
    strip_prefix = "toolchains_llvm-1ca11e2755544e5305d442d72018234eca914604",
    url = "https://github.com/bazel-contrib/toolchains_llvm/archive/1ca11e2755544e5305d442d72018234eca914604.tar.gz",
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
