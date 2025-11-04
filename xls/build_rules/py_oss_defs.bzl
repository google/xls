# Copyright 2025 The XLS Authors
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

"""Shim layer for OSS to redirect Google-internal Python rules to standard rules_python versions."""

load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

py_strict_binary = py_binary
pytype_binary = py_binary
pytype_strict_binary = py_binary

py_strict_library = py_library
pytype_library = py_library
pytype_strict_library = py_library

py_strict_test = py_test
pytype_contrib_test = py_test
pytype_strict_contrib_test = py_test
