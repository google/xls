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

"""Provides helper that initializes external repositories with third-party code."""

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
load("@com_google_benchmark//:bazel/benchmark_deps.bzl", "benchmark_deps")
load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
load("@com_grail_bazel_compdb//:deps.bzl", "bazel_compdb_deps")
load("@io_bazel_rules_closure//closure:repositories.bzl", "rules_closure_dependencies", "rules_closure_toolchains")
load("@python39//:defs.bzl", python_interpreter_target = "interpreter")
load("@rules_7zip//:setup.bzl", "setup_7zip")  # needed by rules_hdl
load("@rules_hdl//:init.bzl", rules_hdl_init = "init")
load("@rules_hdl//dependency_support:dependency_support.bzl", rules_hdl_dependency_support = "dependency_support")
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")
load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
load("@rules_python//python:pip.bzl", "pip_parse")
load("@rules_python//python:repositories.bzl", "py_repositories")
load("//dependency_support/boost:initialize.bzl", initialize_boost = "initialize")
load("//dependency_support/llvm:initialize.bzl", initialize_llvm = "initialize")

def initialize_external_repositories():
    """Calls set-up methods for external repositories that require that."""
    bazel_skylib_workspace()
    protobuf_deps()
    rules_hdl_init(python_interpreter_target = python_interpreter_target)
    rules_hdl_dependency_support()
    setup_7zip()
    rules_closure_dependencies()
    rules_closure_toolchains()
    rules_proto_dependencies()
    rules_proto_toolchains()
    py_repositories()
    pip_parse(
        name = "xls_pip_deps",
        requirements_lock = "//dependency_support:pip_requirements_lock.txt",
        python_interpreter_target = python_interpreter_target,
        timeout = 600000,
    )
    initialize_boost()
    initialize_llvm()
    bazel_compdb_deps()
    benchmark_deps()
    rules_pkg_dependencies()
