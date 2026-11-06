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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("//xls/build_rules:genrule_wrapper.bzl", "genrule_wrapper")
load("//xls/build_rules:xls_cc_embed_data_rules.bzl", "xls_cc_embed_data")
load("//xls/build_rules:xls_diff_test.bzl", "diff_test")
load("//xls/build_rules:xls_utilities.bzl", "proto_data")

def _generate_area_model_registration_impl(ctx):
    output_file = ctx.actions.declare_file(ctx.attr.name + ".cc")
    header_path = ctx.attr.data_lib[CcInfo].compilation_context.direct_public_headers[0].path
    ctx.actions.expand_template(
        template = ctx.file._registration_template,
        output = output_file,
        substitutions = {
            "{WARNING}": "Generated file. Do not edit.",
            "{HEADER}": header_path,
            "{MODEL_NAME}": ctx.attr.model_name,
            "{ACCESS_FN}": ctx.attr.access_fn,
        },
    )
    return [DefaultInfo(files = depset([output_file]))]

_generate_area_model_registration = rule(
    implementation = _generate_area_model_registration_impl,
    attrs = {
        "model_name": attr.string(mandatory = True),
        "access_fn": attr.string(mandatory = True),
        "data_lib": attr.label(
            mandatory = True,
            providers = [CcInfo],
        ),
        "_registration_template": attr.label(
            default = Label("//xls/estimators/area_model:area_model_registration.cc.tmpl"),
            allow_single_file = True,
        ),
    },
    doc = "Generates the C++ source file registering a pre-computed proto area model.",
)

def area_model(
        name,
        model_name,
        srcs,
        precomputed_proto = None,
        **kwargs):
    """Generates an area model cc_library from an EstimatorModel protobuf.

    Args:

      name: Name of the cc_library target to generate.
      model_name: Name of the model. This is the string that is used to access
        the model when calling xls::GetAreaEstimator.
      srcs: The pbtext file containing the EstimatorModel proto. There should only
        be a single source file.
      precomputed_proto: Optional pre-computed CompiledEstimatorModel textproto
        file. If provided, the cc_library embeds the binary proto directly and
        evaluates it at runtime without depending on Python curve-fitting tools,
        and a diff_test target is created to verify it stays in sync with srcs.
      **kwargs: Keyword args to pass to cc_library and genrule_wrapper rules.
    """

    if len(srcs) != 1:
        fail("More than one source not currently supported.")

    precomputed_gen_target = "{}_precomputed_gen".format(name)
    binpb_target = "{}_binpb".format(name)
    data_target = "{}_area_data".format(name)
    source_target = "{}_source".format(name)
    testonly = kwargs.get("testonly", False)

    genrule_wrapper(
        name = precomputed_gen_target,
        srcs = srcs,
        outs = ["{}_generated_precomputed.textproto".format(name)],
        cmd = "$(location //xls/estimators:generate_compiled_estimator_model) $< > $(OUTS)",
        tools = [
            "//xls/estimators:generate_compiled_estimator_model",
        ],
        testonly = testonly,
        visibility = ["//visibility:private"],
    )

    if precomputed_proto != None:
        precomputed_src = precomputed_proto
        diff_test(
            name = "{}_precomputed_up_to_date_test".format(name),
            file = ":{}".format(precomputed_gen_target),
            golden = precomputed_proto,
            tags = [
                "noasan",
                "nomsan",
            ],
        )
    else:
        precomputed_src = ":{}".format(precomputed_gen_target)

    proto_data(
        name = binpb_target,
        src = precomputed_src,
        proto_name = "xls.estimator_model.CompiledEstimatorModel",
        testonly = testonly,
        visibility = ["//visibility:private"],
    )
    xls_cc_embed_data(
        name = data_target,
        data = ":{}".format(binpb_target),
        namespace = "xls::area_model",
        testonly = testonly,
        visibility = ["//visibility:private"],
    )
    _generate_area_model_registration(
        name = source_target,
        model_name = model_name,
        access_fn = "::xls::area_model::get_{}".format(data_target),
        data_lib = ":{}".format(data_target),
        testonly = testonly,
        visibility = ["//visibility:private"],
    )
    cc_library(
        name = name,
        srcs = [":{}".format(source_target)],
        alwayslink = 1,
        deps = [
            ":{}".format(data_target),
            "@abseil-cpp//absl/log:check",
            "//xls/common:module_initializer",
            "//xls/estimators/area_model:area_estimator",
        ],
        **kwargs
    )
