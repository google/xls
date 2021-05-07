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

# XLS-internal-only.
package_group(
    name = "xls_internal",
    packages = [
        "//xls/...",
        "//third_party/xls_berkeley_softfloat/...",
        "//third_party/xls_go_math/...",
        "//third_party/xls_machsuite/...",
    ],
)

# TODO(leary): 2020-06-29 This structure is a work-around to publically expose
# XLS APIs. Once the public XLS API is determined, those packages should be
# explicitly marked with 'xls_public' visibility, and this should be changed so
# that 'xls_public' includes 'xls_internal' and not vice versa.
package_group(
    name = "xls_users",
    # Everything users can see is also visible internally.
    includes = [":xls_internal"],
    packages = ["//..."],
)

# Users of various XLS utility helpers, e.g. things like logging, testing
# targets. Some of these will likely be eliminated when logging is
# no longer XLS specific.
#
# We call this out to distinguish it from properly public headers that expose
# fundamental XLS functionality.
package_group(
    name = "xls_utility_users",
    # Everything utility users can see is also visible internally.
    includes = [":xls_internal"],
    packages = ["//..."],
)

# "Best effort" users are ones that are using some XLS internals
# (ideally temporarily) and are agreeing to be broken at any
# time if we can't figure out how to fix them, since we need to
# be able to evolve XLS innards as necessary.
package_group(
    name = "xls_best_effort_users",
    # Everything best effort users can see is also visible internally.
    includes = [":xls_internal"],
    packages = ["//..."],
)

package(
    default_visibility = [":xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])
