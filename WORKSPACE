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

# TODO: https://github.com/google/xls/issues/931 - Everything here should go away and migrate to MODULE.bazel

workspace(name = "com_google_xls")

load("//dependency_support:load_external.bzl", "load_external_repositories")

load_external_repositories()

# This is still needed while we have some WORKSPACE dependencies that call rules_python.
# Otherwise, you get errors about missing @rules_python_internal in WORKSPACE.
load(
    "@rules_python//python:repositories.bzl",
    "py_repositories",
)

# Must be called before using anything from rules_python, namely in initialize_external_repositories.
# https://github.com/bazelbuild/rules_python/issues/1560#issuecomment-1815118394
py_repositories()

load("//dependency_support:initialize_external.bzl", "initialize_external_repositories")

initialize_external_repositories()
