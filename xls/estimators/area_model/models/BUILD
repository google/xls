# Copyright 2024 The XLS Authors
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

# Delay models.

# @rules_cc//cc:cc_library.bzl
load("//xls/estimators/area_model:build_defs.bzl", "area_model")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# A srcs-less library which gathers together the models into a single
# dependency.
cc_library(
    name = "models",
    srcs = [],
    deps = [
        ":area_model_asap7",
        ":area_model_sky130",
        ":area_model_unit",  # build_cleaner: keep
    ],
)

area_model(
    name = "area_model_asap7",
    srcs = ["asap7.textproto"],
    model_name = "asap7",
)

area_model(
    name = "area_model_sky130",
    srcs = ["sky130.textproto"],
    model_name = "sky130",
)

area_model(
    name = "area_model_unit",
    srcs = ["unit.textproto"],
    model_name = "unit",
)
