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

_LLVM_BINS = [
    "clang",
    "clang-format",
]

_LLVM_PREFIXED_BINS = ["bin/" + bin for bin in _LLVM_BINS]

genrule(
    name = "link_clang_bins",
    outs = _LLVM_PREFIXED_BINS,
    cmd = """
      for i in $(OUTS); do
        f=$${i#$(@D)}
        mkdir -p $(@D)
        ln -sf $$(which $$f) $(@D)/$$f
      done
    """,
    visibility = ["//visibility:public"],
)
