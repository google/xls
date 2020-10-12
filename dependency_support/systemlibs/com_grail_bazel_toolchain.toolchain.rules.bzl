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

def _llvm_toolchain_impl():
    pass

llvm_toolchain = repository_rule(
    attrs = {
        "llvm_version": attr.string(
            default = "6.0.0",
            doc = "One of the supported versions of LLVM.",
        ),
    },
    local = False,
    implementation = _llvm_toolchain_impl,
)
