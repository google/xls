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

"""
This module contains internal build definitions for XLS.

This is only for OSS use.
"""

visibility([
    "//xls/build_rules",
    "//xls/common",
    "//xls/jit",
    "//xls/tools",
])

# Should the AOT code consider us as being MSANd.
# TODO(allight): 2023-12-18: Make this be set automatically somehow?
XLS_IS_MSAN_BUILD = False

# Platform-specific srcs/deps for common targets.
XLS_TIMEOUT_SUPPORT_SRCS = ["timeout_support.cc"]

XLS_TIMEOUT_SUPPORT_EXTRA_DEPS = []

XLS_MODULE_INITIALIZER_SRCS = ["module_initializer.inc"]

XLS_MODULE_INITIALIZER_DEPS = []

XLS_THREAD_SRCS = ["thread.inc"]

XLS_THREAD_DEPS = []

XLS_INIT_XLS_SRCS = ["init_xls.cc"]

XLS_INIT_XLS_EXTRA_HDRS = ["build_embed.h"]

XLS_INIT_XLS_DATA = []

XLS_INIT_XLS_LINKSTAMP = "build_embed.cc"

XLS_INIT_XLS_EXTRA_DEPS = [
    "@abseil-cpp//absl/flags:config",
    "@abseil-cpp//absl/flags:parse",
    "@abseil-cpp//absl/flags:usage",
    "@abseil-cpp//absl/log:initialize",
]

# Optional runtime/diagnostics deps for tool binaries.
XLS_BINARY_STACKTRACE_DEPS = []

XLS_BINARY_HEAP_PROFILER_DEPS = []

XLS_RPC_CLIENT_EXTRA_DEPS = []
