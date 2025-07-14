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

"""Contains macros for creating XLS area models."""

# @rules_cc//cc:cc_library.bzl
load("//xls/build_rules:genrule_wrapper.bzl", "genrule_wrapper")

def area_model(
        name,
        model_name,
        srcs,
        **kwargs):
    """Generates an area model cc_library from an EstimatorModel protobuf.

    Args:

      name: Name of the cc_library target to generate.
      model_name: Name of the model. This is the string that is used to access
        the model when calling xls::GetDelayEstimator.
      srcs: The pbtext file containing the EstimatorModel proto. There should only
        be a single source file.
      **kwargs: Keyword args to pass to cc_library and genrule_wrapper rules.
    """

    if len(srcs) != 1:
        fail("More than one source not currently supported.")

    genrule_wrapper(
        name = "{}_source".format(name),
        srcs = srcs,
        outs = ["{}.cc".format(name)],
        cmd = ("$(location //xls/estimators/area_model:generate_area_lookup) " +
               "--model_name={model_name} $< " +
               "| $(location @llvm_toolchain//:clang-format)" +
               " > $(OUTS)").format(model_name = model_name),
        tools = [
            "//xls/estimators/area_model:generate_area_lookup",
            "@llvm_toolchain//:clang-format",
        ],
        **kwargs
    )
    native.cc_library(
        name = name,
        srcs = [":{}_source".format(name)],
        alwayslink = 1,
        deps = [
            "@com_google_absl//absl/container:flat_hash_set",
            "@com_google_absl//absl/log:check",
            "@com_google_absl//absl/memory",
            "@com_google_absl//absl/status",
            "//xls/common:module_initializer",
            "@com_google_absl//absl/status:statusor",
            "//xls/estimators/area_model:area_estimator",
            "//xls/ir",
        ],
        **kwargs
    )
