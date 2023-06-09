# Copyright 2023 The XLS Authors
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

load("@rules_python//python:pip.bzl", "pip_install")

"""Initializes Renode."""

def renode_register_toolchain():
    native.register_toolchains(
        "@renode//:renode_linux_toolchain",
    )

def initialize():
    renode_register_toolchain()
    pip_install(
        name = "renode_robot_deps",
        requirements = "@renode//:tests/requirements.txt",
        python_interpreter = "python3",
        timeout = 600000,
    )
