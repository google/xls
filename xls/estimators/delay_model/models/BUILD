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

# Delay models.

load(
    "//xls/build_rules:xls_estimator_rules.bzl",
    "estimator_model_group",
    "xls_default_delay_models",
)
load("//xls/estimators/delay_model:build_defs.bzl", "delay_model")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# Alias for whatever the toolchain says the available area models are.
xls_default_delay_models(name = "models")

estimator_model_group(
    name = "oss_models",
    models = [
        ":model_asap7",
        ":model_sky130",
        ":model_unit",
    ],
)

delay_model(
    name = "model_unit",
    srcs = ["unit.textproto"],
    model_name = "unit",
    precedence = "kLow",
)

delay_model(
    name = "model_asap7",
    srcs = ["asap7.textproto"],
    model_name = "asap7",
    precedence = "kMedium",
)

delay_model(
    name = "model_sky130",
    srcs = ["sky130.textproto"],
    model_name = "sky130",
    precedence = "kMedium",
)
