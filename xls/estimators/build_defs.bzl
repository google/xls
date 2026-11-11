# Copyright 2026 The XLS Authors
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

"""Shared Starlark helpers for XLS delay and area estimator model rules."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("//xls/build_rules:genrule_wrapper.bzl", "genrule_wrapper")
load("//xls/build_rules:xls_cc_embed_data_rules.bzl", "xls_cc_embed_data")
load("//xls/build_rules:xls_diff_test.bzl", "diff_test")
load("//xls/build_rules:xls_utilities.bzl", "proto_data")

def _generate_estimator_model_registration_impl(ctx):
    output_file = ctx.actions.declare_file(ctx.attr.name + ".cc")
    header_path = ctx.attr.data_lib[CcInfo].compilation_context.direct_public_headers[0].short_path
    substitutions = {
        "{WARNING}": "Generated file. Do not edit.",
        "{HEADER}": header_path,
        "{MODEL_NAME}": ctx.attr.model_name,
        "{ACCESS_FN}": ctx.attr.access_fn,
    }
    substitutions.update(ctx.attr.extra_substitutions)
    ctx.actions.expand_template(
        template = ctx.file.registration_template,
        output = output_file,
        substitutions = substitutions,
    )
    return [DefaultInfo(files = depset([output_file]))]

generate_estimator_model_registration = rule(
    implementation = _generate_estimator_model_registration_impl,
    attrs = {
        "model_name": attr.string(mandatory = True),
        "access_fn": attr.string(mandatory = True),
        "data_lib": attr.label(
            mandatory = True,
            providers = [CcInfo],
        ),
        "registration_template": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "extra_substitutions": attr.string_dict(),
    },
    doc = "Generates the C++ source file registering a pre-computed proto estimator model.",
)

def estimator_model_cc_library(
        name,
        model_name,
        srcs,
        embed_namespace,
        data_target_suffix,
        registration_template,
        estimator_dep,
        precomputed_proto = None,
        extra_substitutions = None,
        **kwargs):
    """Generates an estimator model cc_library from an EstimatorModel protobuf.

    Args:
      name: Name of the cc_library target to generate.
      model_name: Name of the model registered in the estimator registry.
      srcs: Single-element list containing the EstimatorModel textproto file.
      embed_namespace: C++ namespace in which to embed the binary proto data.
      data_target_suffix: Suffix appended to `name` for the embedded data target.
      registration_template: Label of the C++ registration `.cc.tmpl` file.
      estimator_dep: Label of the C++ estimator library (`delay_estimator` or
        `area_estimator`).
      precomputed_proto: Optional pre-computed CompiledEstimatorModel textproto
        file. If provided, a `diff_test` verifies it stays in sync with `srcs`.
      extra_substitutions: Optional dict of additional template substitutions.
      **kwargs: Keyword args forwarded to `cc_library`.
    """
    if len(srcs) != 1:
        fail("More than one source not currently supported.")

    precomputed_gen_target = "{}_precomputed_gen".format(name)
    binpb_target = "{}_binpb".format(name)
    data_target = "{}_{}".format(name, data_target_suffix)
    source_target = "{}_source".format(name)
    testonly = kwargs.get("testonly", False)

    genrule_srcs = list(srcs)
    genrule_cmd = "$(execpath //xls/estimators:generate_compiled_estimator_model) $(execpath {})".format(srcs[0])
    if precomputed_proto != None:
        genrule_srcs.append(precomputed_proto)
        genrule_cmd += " --reference_golden=$(execpath {})".format(precomputed_proto)
    genrule_cmd += " > $(OUTS)"

    genrule_wrapper(
        name = precomputed_gen_target,
        srcs = genrule_srcs,
        outs = ["{}_generated_precomputed.textproto".format(name)],
        cmd = genrule_cmd,
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
        namespace = embed_namespace,
        testonly = testonly,
        visibility = ["//visibility:private"],
    )
    generate_estimator_model_registration(
        name = source_target,
        model_name = model_name,
        access_fn = "::{}::get_{}".format(embed_namespace, data_target),
        data_lib = ":{}".format(data_target),
        registration_template = registration_template,
        extra_substitutions = extra_substitutions or {},
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
            estimator_dep,
        ],
        **kwargs
    )
