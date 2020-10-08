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

# XLS: Accelerator Synthesis

# TODO(leary): 2020-06-29 This structure is a work-around to publically expose
# XLS APIs. Once the public XLS API is determined, those packages should be
# explicitly marked with 'xls_public' visibility, and this should be changed so
# that 'xls_public' includes 'xls_internal' and not vice versa.
package_group(
    name = "xls_public",
    packages = ["//..."],
)

package_group(
    name = "xls_internal",
    includes = [":xls_public"],
    packages = ["//xls/..."],
)

package(
    default_visibility = [":xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])
