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
load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
load("@io_bazel_rules_closure//closure:repositories.bzl", "rules_closure_dependencies", "rules_closure_toolchains")
load("@rules_python//python:pip.bzl", "pip_import")
load("//dependency_support/boost:initialize.bzl", initialize_boost = "initialize")

def initialize_external_repositories():
    """Calls set-up methods for external repositories that require that."""
    bazel_skylib_workspace()
    protobuf_deps()
    python_configure(name = "local_config_python")
    rules_closure_dependencies()
    rules_closure_toolchains()
    pip_import(
        name = "xls_pip_deps",
        requirements = "//dependency_support:pip_requirements.txt",
        python_interpreter = "python3",
        timeout = 600000,
    )
    initialize_boost()
