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
# LINT.IfChange

# XLS: Accelerator Synthesis

# The group of known users of XLS.
# The targets listed with this visibility comprise the intended "public API" of
# XLS, and are be supported and are "reasonably" stable, i.e., significant
# changes are expected to be rare and coordinated.
# Ultimately, this is where all non-XLS-internal users should be listed, once
# the transition described in the TODOs below are complete.
package_group(
    name = "xls_public",
    # Everything the public can see is also visible internally.
    includes = [":xls_internal"],
    packages = ["//..."],
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

# XLS MLIR dialect users.
package_group(
    name = "xls_mlir_users",
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

# XLS-internal-only.
# Ideally only XLS internal implementation would be present in here.
package_group(
    name = "xls_internal",
    packages = ["//..."],
)
# LINT.ThenChange(//xls/BUILD)
