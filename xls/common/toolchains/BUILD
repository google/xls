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

package(
    # Avoid base BUILD file since it pulls in extra things.
    default_applicable_licenses = ["//xls/common/toolchains/license:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

toolchain_type(name = "toolchain_type")

toolchain(
    name = "xls_default_configuration_toolchain",
    toolchain = "//xls/common/toolchains/config:default_configuration",
    toolchain_type = ":toolchain_type",
)
