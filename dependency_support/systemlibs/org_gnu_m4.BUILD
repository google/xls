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

licenses(["restricted"])  # GPLv3

genrule(
    name = "link_m4",
    outs = ["bin/m4"],
    cmd = "ln -sf $$(which m4) $@",
)

sh_binary(
    name = "m4",
    srcs = ["bin/m4"],
    visibility = ["//visibility:public"],
)
