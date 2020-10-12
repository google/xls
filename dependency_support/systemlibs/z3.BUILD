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

licenses(["notice"]) # MIT license

package(default_visibility = ["//visibility:public"])

_Z3_HEADERS = [
    "z3.h",
    "z3_algebraic.h",
    "z3_api.h",
    "z3_ast_containers.h",
    "z3_fixedpoint.h",
    "z3_fpa.h",
    "z3++.h",
    "z3_macros.h",
    "z3_optimization.h",
    "z3_polynomial.h",
    "z3_rcf.h",
    "z3_spacer.h",
    "z3_v1.h",
    "z3_version.h",
]

_Z3_PREFIXED_HEADERS = ["src/api/" + hdr for hdr in _Z3_HEADERS]

# In order to limit the damage from the `includes` propagation
# via `:z3`, copy the public headers to a subdirectory and
# expose those.
genrule(
    name = "copy_public_headers",
    outs = _Z3_PREFIXED_HEADERS,
    cmd = """
      for i in $(OUTS); do
        f=$${i#$(@D)/src/api/}
        mkdir -p $(@D)
        ln -sf /usr/include/z3/$$f $(@D)/src/api/$$f
      done
    """,
    visibility = ["//visibility:private"],
)

cc_library(
    name = "z3lib",
    linkopts = ["-lz3"],
)

cc_library(
    name = "api",
    hdrs = _Z3_PREFIXED_HEADERS,
    deps = [":z3lib"],
)
