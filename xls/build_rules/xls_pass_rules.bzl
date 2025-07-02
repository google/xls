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

"""This module contains the rules for defining xls passes."""

load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")
load(
    "//xls/build_rules:xls_providers.bzl",
    "XlsOptimizationPassInfo",
    "XlsOptimizationPassRegistryInfo",
)

# Load build tooling macros
def register_extension_info(*args, **kwargs):
    return None

def _generate_registration_impl(ctx):
    output_file = ctx.actions.declare_file(ctx.attr.name + ".cc")
    headers = ""
    for h in ctx.attr.lib[CcInfo].compilation_context.direct_public_headers:
        headers += "\n"
        headers += '#include "%s"' % h.path
    ctx.actions.expand_template(
        template = ctx.file._registration_template,
        output = output_file,
        substitutions = {
            "{REGISTRATION_NAME}": ctx.attr.name,
            "{SHORT_NAME}": ctx.attr.pass_class + "::kName",
            "{NAME}": ctx.attr.pass_class,
            "{WARNING}": "Generated file. Do not edit.",
            "{HEADERS}": headers,
        },
    )

    return [DefaultInfo(files = depset([output_file]))]

_generate_registration = rule(
    implementation = _generate_registration_impl,
    attrs = {
        "_registration_template": attr.label(
            default = Label("//xls/passes/tools:pass_registration.cc.tmpl"),
            allow_single_file = True,
        ),
        "pass_class": attr.string(
            doc = "String name (with namespace) of the pass being created.",
            mandatory = True,
        ),
        "lib": attr.label(
            doc = "library we are generating for",
            mandatory = True,
            providers = [CcInfo],
        ),
    },
    doc = "Helper rule to perform source code generation for registering passes.",
)

def _xls_pass_and_registration_impl(ctx):
    return [
        # If used as a dep default to just linking the pass itself.
        ctx.attr.pass_lib[CcInfo],
        # On cmd line just build as though this was the pass impl.
        ctx.attr.pass_lib[DefaultInfo],
        # If used in pass_registry note the registration library.
        XlsOptimizationPassInfo(
            pass_impl = ctx.attr.pass_lib,
            pass_registration = ctx.attr.pass_reg,
        ),
    ]

_xls_pass_and_registration = rule(
    implementation = _xls_pass_and_registration_impl,
    doc = """Helper rule to note the actual pass and registration libraries in a provider.

        This lets it be used in both cc-library deps positions and in the pass_registry.""",
    attrs = {
        "pass_lib": attr.label(
            doc = "library we are generating for",
            mandatory = True,
            providers = [CcInfo],
        ),
        "pass_reg": attr.label(
            doc = "library we are generating for",
            mandatory = True,
            providers = [CcInfo],
        ),
    },
)

# TODO(allight): We could make this a 'rule' but we need a macro solely to make
# sure that build-rule automation still works so might as well leave it as a
# macro.
# TODO(allight): Use new macro() system when available.
def xls_pass(name, pass_class, tags = [], **kwargs):
    """A rule to build an xls pass library and the registration library.

    Arguments are the ones from cc_library with pass_class denoting the class name of the pass.

    This should be used with `xls_pass_registry` to ensure the pass is available for injection.

    The xls_pass_registry will automatically select the registration library when used.

    Example:

    ```
    xls_pass(
      name = "reassociation_pass",
      pass_class = "xls::ReassociationPass",
      srcs = ["reassociation_pass.cc"],
      hdrs = ["reassociation_pass.h"],
      deps = [
        ...
      ],
    )
    ```

    Args:
      name: The name of the pass library.
      pass_class: String name (with namespace) of the pass being created.
      tags: Any tags to put on generated libraries.
      **kwargs: Keyword arguments to pass to the cc_library.
    """

    # Ensure only the pass-and-registration target has a visibility.
    if "visibility" in kwargs:
        # The internal pass-target should not be visible. Only the external
        # pass-and-reg provider should be potentially public.
        visibility = kwargs["visibility"]
        kwargs.pop("visibility")
    else:
        visibility = None

    # Force build-automation stuff to choose the pass-and-reg target for direct users.
    impl_tags = tags + ["alt_dep=%s" % name, "avoid_dep"]
    native.cc_library(
        name = "%s_impl" % name,
        visibility = ["//visibility:private"],
        tags = impl_tags,
        **kwargs
    )
    _generate_registration(
        name = "%s_registration_cc" % name,
        lib = "%s_impl" % name,
        visibility = ["//visibility:private"],
        pass_class = pass_class,
    )
    native.cc_library(
        name = "%s_registration" % name,
        srcs = ["%s_registration_cc" % name],
        visibility = ["//visibility:private"],
        tags = tags,
        deps = [
            "%s_impl" % name,
            "//xls/passes:optimization_pass_registry",
            "//xls/common:module_initializer",
            "@com_google_absl//absl/log:check",
        ],
        alwayslink = True,
    )
    _xls_pass_and_registration(
        name = name,
        pass_lib = "%s_impl" % name,
        pass_reg = "%s_registration" % name,
        visibility = visibility,
    )

register_extension_info(
    extension = xls_pass,
    label_regex_for_dep = "{extension_name}_impl",
)

def _xls_pass_registry_impl(ctx):
    inputs = []
    passes = []
    for c in ctx.attr.passes:
        if XlsOptimizationPassInfo in c:
            inputs.append(c[XlsOptimizationPassInfo].pass_registration[CcInfo].linking_context)
            passes.append(c[XlsOptimizationPassInfo].pass_registration)
        elif XlsOptimizationPassRegistryInfo in c:
            for x in c[XlsOptimizationPassRegistryInfo].passes:
                inputs.append(x[CcInfo].linking_context)
            passes.extend(c[XlsOptimizationPassRegistryInfo].passes)
        else:
            inputs.append(c[CcInfo].linking_context)
            passes.append(c[CcInfo])

    # For now we don't compile anything.
    comp_out = cc_common.create_compilation_outputs()
    comp_ctx = cc_common.create_compilation_context()
    (link, out) = cc_common.create_linking_context_from_compilation_outputs(
        name = ctx.label.name,
        actions = ctx.actions,
        feature_configuration = cc_common.configure_features(
            ctx = ctx,
            cc_toolchain = find_cpp_toolchain(ctx),
            requested_features = ctx.features,
            unsupported_features = ctx.disabled_features,
        ),
        cc_toolchain = find_cpp_toolchain(ctx),
        compilation_outputs = comp_out,
        linking_contexts = inputs,
        # This target links in all the passes.
        alwayslink = True,
    )
    return [
        CcInfo(compilation_context = comp_ctx, linking_context = link),
        XlsOptimizationPassRegistryInfo(passes = inputs),
        DefaultInfo(files = depset(
            # Ensure just building the registry does force all the passes to build.
            direct = out.library_to_link.pic_objects,
            transitive = [c[DefaultInfo].files for c in ctx.attr.passes],
        )),
    ]

xls_pass_registry = rule(
    implementation = _xls_pass_registry_impl,
    doc = "Registry library that registers the given passes.",
    provides = [CcInfo, XlsOptimizationPassRegistryInfo],
    fragments = ["cpp"],
    toolchains = use_cpp_toolchain(),
    attrs = {
        "passes": attr.label_list(
            doc = "List of pass infos, libraries or other pass_registry rules to register.",
            providers = [[XlsOptimizationPassRegistryInfo], [XlsOptimizationPassInfo], [CcInfo]],
        ),
    },
)
