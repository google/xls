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

"""Adapter between open source and Google-internal py_proto_library rules."""

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

def xls_py_proto_library(name, internal_deps, srcs, deps = []):
    py_proto_library(
        name = name,
        srcs = srcs,
        deps = deps,
    )
