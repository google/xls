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

"""Contains macros for creating XLS delay models."""

load("//xls/estimators:build_defs.bzl", "estimator_model_cc_library")

_EMBED_NAMESPACE = "xls::delay_model"

def delay_model(
        name,
        model_name,
        precedence,
        srcs,
        precomputed_proto = None,
        **kwargs):
    """Generates a delay model cc_library from an EstimatorModel protobuf.

    Args:

      name: Name of the cc_library target to generate.
      model_name: Name of the model. This is the string that is used to access
        the model when calling xls::GetDelayEstimator.
      precedence: Precedence for the model in the delay model registry.
      srcs: The pbtext file containing the EstimatorModel proto. There should only
        be a single source file.
      precomputed_proto: Optional pre-computed CompiledEstimatorModel textproto
        file. If provided, the cc_library embeds the binary proto directly and
        evaluates it at runtime without depending on Python curve-fitting tools,
        and a diff_test target is created to verify it stays in sync with srcs.
      **kwargs: Keyword args to pass to cc_library and genrule_wrapper rules.
    """
    if precedence not in ("kLow", "kMedium", "kHigh"):
        fail("Invalid precedence for delay model: " + precedence)

    estimator_model_cc_library(
        name = name,
        model_name = model_name,
        srcs = srcs,
        precomputed_proto = precomputed_proto,
        embed_namespace = _EMBED_NAMESPACE,
        data_target_suffix = "delay_data",
        registration_template = Label("//xls/estimators/delay_model:delay_model_registration.cc.tmpl"),
        estimator_dep = "//xls/estimators/delay_model:delay_estimator",
        extra_substitutions = {"{PRECEDENCE}": precedence},
        **kwargs
    )
