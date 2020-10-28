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

"""Wraps the pybind_extension rule with a py_library rule."""

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

def xls_pybind_extension(**kwargs):
    name = kwargs["name"]
    py_deps = kwargs.pop("py_deps", [])
    pybind_extension(
        # We want our Python extension sharedlibs to resolve their symbols
        # against the main binary; otherwise very odd effects will be observed
        # due to the duplication of rodata; e.g. the absl hash routines rely on
        # a symbol's address to be randomized via ASLR for use as a seed!
        linkstatic = False,
        **kwargs
    )
    native.py_library(
        name = name,
        data = [name + ".so"],
        deps = py_deps,
    )
