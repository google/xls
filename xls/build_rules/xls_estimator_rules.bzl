# Copyright 2025 The XLS Authors
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

"""This module contains the rules for defining xls delay models."""

load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load(
    "//xls/build_rules:xls_providers.bzl",
    "XlsEstimatorModelInfo",
    "XlsEstimatorRegistryInfo",
)

def _estimator_model_group(ctx):
    models = []
    for f in ctx.attr.models:
        if XlsEstimatorRegistryInfo in f:
            models.extend(f[XlsEstimatorRegistryInfo].models)
        else:
            models.append(XlsEstimatorModelInfo(cc_info = f[CcInfo], default_info = f[DefaultInfo]))
    return [
        XlsEstimatorRegistryInfo(models = models),
        DefaultInfo(
            files = depset(
                direct = [],
                transitive = [
                    f[DefaultInfo].files
                    for f in ctx.attr.models
                ],
            ),
        ),
    ]

estimator_model_group = rule(
    implementation = _estimator_model_group,
    doc = """A group of delay models.""",
    provides = [XlsEstimatorRegistryInfo],
    attrs = {
        "models": attr.label_list(
            doc = "List of models or other model_groups",
            mandatory = True,
            providers = [[CcInfo, DefaultInfo], [XlsEstimatorRegistryInfo, DefaultInfo]],
        ),
    },
)

def _xls_default_estimator_models(ctx, reg):
    models = reg.models
    cc_toolchain = find_cpp_toolchain(ctx)
    cc_features = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    linking_ctxs = []
    all_runfiles = []
    for model in models:
        linking_ctxs.append(model.cc_info.linking_context)
        all_runfiles.append(model.default_info.default_runfiles)
    rf = ctx.runfiles().merge_all(all_runfiles)
    comp_out = cc_common.create_compilation_outputs()
    comp_ctx = cc_common.create_compilation_context()
    (link, out) = cc_common.create_linking_context_from_compilation_outputs(
        name = ctx.label.name,
        actions = ctx.actions,
        feature_configuration = cc_features,
        cc_toolchain = cc_toolchain,
        compilation_outputs = comp_out,
        linking_contexts = linking_ctxs,
        # This target links in all the passes.
        alwayslink = True,
    )
    cc_info = CcInfo(compilation_context = comp_ctx, linking_context = link)
    default_info = DefaultInfo(
        files = depset(
            # Ensure just building the registry does force all the passes to build.
            direct = out.library_to_link.pic_objects,
            transitive = [c.default_info.files for c in models],
        ),
        runfiles = rf,
    )
    return [reg, cc_info, default_info]

def _xls_default_delay_models(ctx):
    reg = ctx.toolchains["//xls/common/toolchains:toolchain_type"].configuration.delay_model_registry
    return _xls_default_estimator_models(ctx, reg)

_xls_default_delay_models_rule = rule(
    implementation = _xls_default_delay_models,
    doc = """A library which includes the default delay models""",
    provides = [XlsEstimatorRegistryInfo, CcInfo],
    toolchains = use_cpp_toolchain() + ["//xls/common/toolchains:toolchain_type"],
    fragments = ["cpp"],
)

def xls_default_delay_models(name, tags = None):
    """A library which includes the default delay models.

    Args:
      name: The name of the target.
      tags: Tags to apply to the target.
    """
    if tags == None:
        tags = ["keep_dep"]
    elif "keep_dep" not in tags:
        tags = tags + ["keep_dep"]
    _xls_default_delay_models_rule(
        name = name,
        tags = tags,
    )

def _xls_default_area_models(ctx):
    reg = ctx.toolchains["//xls/common/toolchains:toolchain_type"].configuration.area_model_registry
    return _xls_default_estimator_models(ctx, reg)

_xls_default_area_models_rule = rule(
    implementation = _xls_default_area_models,
    doc = """A library which includes the default area models""",
    provides = [XlsEstimatorRegistryInfo, CcInfo],
    toolchains = use_cpp_toolchain() + ["//xls/common/toolchains:toolchain_type"],
    fragments = ["cpp"],
)

def xls_default_area_models(name, tags = None):
    """A library which includes the default area models.

    Args:
      name: The name of the target.
      tags: Tags to apply to the target.
    """
    if tags == None:
        tags = ["keep_dep"]
    elif "keep_dep" not in tags:
        tags = tags + ["keep_dep"]
    _xls_default_area_models_rule(
        name = name,
        tags = tags,
    )
