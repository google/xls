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

# TODO(#931): Everything here should go away and migrate to MODULE.bazel

workspace(name = "com_google_xls")

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

load("//dependency_support:initialize_external.bzl", "initialize_external_repositories")

initialize_external_repositories()

load("@xls_pip_deps//:requirements.bzl", xls_pip_install_deps = "install_deps")

xls_pip_install_deps()
