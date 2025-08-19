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

# TODO: https://github.com/google/xls/issues/931 - with MODULE.bazel, probably some of these can be removed now, with the
# eventual goal that none of this is needed anymore and the file can be removed.

load("@project_python//:defs.bzl", python_interpreter_target = "interpreter")
load("@rules_7zip//:setup.bzl", "setup_7zip")  # needed by rules_hdl
load("@rules_hdl//:init.bzl", rules_hdl_init = "init")
load("@rules_hdl//dependency_support:dependency_support.bzl", rules_hdl_dependency_support = "dependency_support")
load("@rules_python//python:pip.bzl", "pip_parse")
load("//dependency_support/llvm:initialize.bzl", initialize_llvm = "initialize")

def initialize_external_repositories():
    """Calls set-up methods for external repositories that require that."""
    rules_hdl_init(python_interpreter_target = python_interpreter_target)
    rules_hdl_dependency_support()
    setup_7zip()
    pip_parse(
        name = "xls_pip_deps",
        requirements_lock = Label("//dependency_support:pip_requirements_lock.txt"),
        python_interpreter_target = python_interpreter_target,
        timeout = 600000,
    )
    initialize_llvm()
