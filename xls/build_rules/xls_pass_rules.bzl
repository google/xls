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

load("@bazel_skylib//lib:dicts.bzl", "dicts")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load(
    "//xls/build_rules:xls_cc_embed_data_rules.bzl",
    _embed_data_attrs = "embed_data_attrs",
    _get_embedded_data = "get_embedded_data",
)
load(
    "//xls/build_rules:xls_providers.bzl",
    "XlsCodegenPassInfo",
    "XlsCodegenPassRegistryConfigInfo",
    "XlsCodegenPassRegistryInfo",
    "XlsOptimizationPassInfo",
    "XlsOptimizationPassRegistryConfigInfo",
    "XlsOptimizationPassRegistryInfo",
)
load(
    "//xls/build_rules:xls_utilities.bzl",
    _proto_data_tool_attrs = "proto_data_tool_attrs",
    _textproto_to_binary = "text_proto_to_binary",
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
        template = ctx.file.registration_template,
        output = output_file,
        substitutions = {
            "{REGISTRATION_NAME}": ctx.attr.name,
            "{SHORT_NAME}": ctx.attr.pass_class + "::kName",
            "{NAME}": ctx.attr.pass_class,
            "{WARNING}": "Generated file. Do not edit.",
            "{HEADERS}": headers,
            "{FIRST_HEADER_FILE}": ctx.attr.lib[CcInfo].compilation_context.direct_public_headers[0].path,
        },
    )

    return [DefaultInfo(files = depset([output_file]))]

_generate_registration = rule(
    implementation = _generate_registration_impl,
    attrs = {
        "registration_template": attr.label(
            allow_single_file = True,
            mandatory = True,
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
            doc = "The library containing the pass registration code.",
            mandatory = True,
            providers = [CcInfo],
        ),
    },
)

def _xls_codegen_pass_and_registration_impl(ctx):
    return [
        ctx.attr.pass_lib[CcInfo],
        ctx.attr.pass_lib[DefaultInfo],
        XlsCodegenPassInfo(
            pass_impl = ctx.attr.pass_lib,
            pass_registration = ctx.attr.pass_reg,
        ),
    ]

_xls_codegen_pass_and_registration = rule(
    implementation = _xls_codegen_pass_and_registration_impl,
    doc = """Helper rule to note the actual codegen pass and registration libraries in a provider.

        This lets it be used in both cc-library deps positions and in the pass_registry.""",
    attrs = {
        "pass_lib": attr.label(
            doc = "library we are generating for",
            mandatory = True,
            providers = [CcInfo],
        ),
        "pass_reg": attr.label(
            doc = "The library containing the pass registration code.",
            mandatory = True,
            providers = [CcInfo],
        ),
    },
)

def _xls_pass_impl_macro(
        name,
        pass_class,
        registration_template,
        registry_dep,
        pass_and_registration_rule,
        tags = [],
        **kwargs):
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
    cc_library(
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
        registration_template = registration_template,
    )
    cc_library(
        name = "%s_registration" % name,
        srcs = ["%s_registration_cc" % name],
        visibility = ["//visibility:private"],
        tags = tags,
        deps = [
            "%s_impl" % name,
            registry_dep,
            "//xls/common:module_initializer",
            "@com_google_absl//absl/log:check",
        ],
        alwayslink = True,
    )
    pass_and_registration_rule(
        name = name,
        pass_lib = "%s_impl" % name,
        pass_reg = "%s_registration" % name,
        visibility = visibility,
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
    _xls_pass_impl_macro(
        name = name,
        pass_class = pass_class,
        registration_template = "//xls/passes/tools:pass_registration.cc.tmpl",
        registry_dep = "//xls/passes:optimization_pass_registry",
        pass_and_registration_rule = _xls_pass_and_registration,
        tags = tags,
        **kwargs
    )

register_extension_info(
    extension = xls_pass,
    label_regex_for_dep = "{extension_name}_impl",
)

def xls_codegen_pass(name, pass_class, tags = [], **kwargs):
    """A rule to build an xls codegen pass library and the registration library.

    Example:
    ```
    xls_codegen_pass(
      name = "clock_gating_pass",
      pass_class = "ClockGatingPass",
      srcs = ["clock_gating_pass.cc"],
      hdrs = ["clock_gating_pass.h"],
      deps = [...],
    )
    ```

    Args:
      name: The name of the pass library.
      pass_class: String name (with namespace) of the pass being created.
      tags: Any tags to put on generated libraries.
      **kwargs: Keyword arguments to pass to the cc_library.
    """
    _xls_pass_impl_macro(
        name = name,
        pass_class = pass_class,
        registration_template = "//xls/codegen_v_1_5/tools:codegen_pass_registration.cc.tmpl",
        registry_dep = "//xls/codegen_v_1_5:codegen_pass_registry",
        pass_and_registration_rule = _xls_codegen_pass_and_registration,
        tags = tags,
        **kwargs
    )

register_extension_info(
    extension = xls_codegen_pass,
    label_regex_for_dep = "{extension_name}_impl",
)

def _xls_pass_registry_impl_helper(
        ctx,
        pass_info_type,
        registry_info_type,
        config_info_type,
        proto_name,
        namespace):
    out_files = []
    if ctx.file.pipeline_binpb and ctx.file.pipeline:
        fail("At most one of pipeline and pipeline_binpb may be present.")
    cc_toolchain = find_cpp_toolchain(ctx)
    cc_features = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    if ctx.file.pipeline_binpb:
        pipeline_binpb = ctx.file.pipeline_binpb
        pipeline_filename = pipeline_binpb.path
    elif ctx.file.pipeline:
        pipeline_filename = ctx.file.pipeline.path
        pipeline_binpb = ctx.actions.declare_file(
            ctx.file.pipeline.basename + "_" + ctx.attr.name + ".binpb",
        )
        out_files.append(pipeline_binpb)
        _textproto_to_binary(
            ctx = ctx,
            src_file = ctx.file.pipeline,
            output_file = pipeline_binpb,
            proto_name = proto_name,
        )
    else:
        pipeline_binpb = None
        pipeline_filename = None

    files_out = []
    inputs = []
    passes = []
    pass_infos = []

    linking_ctxs = []
    existing_binpbs = []
    existing_pipeline_src = []
    for c in ctx.attr.passes:
        if pass_info_type in c:
            inputs.append(c[pass_info_type].pass_registration[CcInfo].linking_context)
            passes.append(c[pass_info_type].pass_registration[CcInfo])
            pass_infos.append(pass_info_type(
                pass_impl = c[pass_info_type].pass_impl[CcInfo],
                pass_registration = c[pass_info_type].pass_registration[CcInfo],
            ))
        elif registry_info_type in c:
            for x in c[registry_info_type].passes:
                inputs.append(x.linking_context)
            passes.extend(c[registry_info_type].passes)
            pass_infos.extend(c[registry_info_type].pass_infos)
            existing_binpbs.append(c[registry_info_type].pipeline_binpb)
            existing_pipeline_src.append(c[registry_info_type].pipeline_src)
        else:
            inputs.append(c[CcInfo].linking_context)
            passes.append(c[CcInfo])
            pass_infos.append(pass_info_type(
                pass_impl = c[CcInfo],
                pass_registration = c[CcInfo],
            ))

    if not pipeline_binpb:
        if len(existing_binpbs) == 1:
            pipeline_binpb = existing_binpbs[0]
            pipeline_filename = existing_pipeline_src[0].path if existing_pipeline_src[0] else "FROM DEPENDENCY"
        elif existing_binpbs:
            fail("Multiple pipeline protos present in dependencies without directly provided pipeline.")

    linking_ctxs.extend(inputs)
    if pipeline_binpb:
        mangled_label = str(abs(hash("@@{}//{}:{}".format(
            ctx.label.workspace_name,
            ctx.label.package,
            ctx.label.name,
        ))))
        accessor = "get_%s_pipeline" % mangled_label
        proto_data_fn = "%s::%s" % (namespace, accessor)
        emb_header = ctx.actions.declare_file("%s_embedded_pipeline.h" % ctx.attr.name)
        emb_cc = ctx.actions.declare_file("%s_embedded_pipeline.cc" % ctx.attr.name)
        files_out.append(emb_header)
        files_out.append(emb_cc)
        embedded_binproto_data = _get_embedded_data(
            ctx = ctx,
            name = ctx.attr.name + "_embedded_pipeline_binpb",
            hdr_file = emb_header,
            cpp_file = emb_cc,
            namespace = namespace,
            accessor = accessor,
            data_file = pipeline_binpb,
        )
        reg_pipeline_cc = ctx.actions.declare_file("%s_declare_pipeline.cc" % ctx.attr.name)
        files_out.append(reg_pipeline_cc)
        ctx.actions.expand_template(
            template = ctx.file._register_pipeline_template,
            output = reg_pipeline_cc,
            substitutions = {
                "{WARNING}": "Generated file. Do not edit",
                "{HEADER}": emb_header.path,
                "{ACCESS_FN}": proto_data_fn,
                "{MANGLED_LABEL}": accessor,
                "{FILE}": pipeline_filename,
            },
        )
        deps = [embedded_binproto_data.compilation_context]
        link_deps = [embedded_binproto_data.linking_context]
        for lib in ctx.attr._register_pipeline_deps:
            deps.append(lib[CcInfo].compilation_context)
            link_deps.append(lib[CcInfo].linking_context)
        (comp_ctx, comp_out) = cc_common.compile(
            name = ctx.label.name + "_pipeline",
            actions = ctx.actions,
            feature_configuration = cc_features,
            cc_toolchain = cc_toolchain,
            srcs = [reg_pipeline_cc],
            compilation_contexts = deps,
        )
        (link_ctx, _link_out) = cc_common.create_linking_context_from_compilation_outputs(
            name = ctx.label.name + "_pipeline",
            actions = ctx.actions,
            feature_configuration = cc_features,
            cc_toolchain = cc_toolchain,
            compilation_outputs = comp_out,
            linking_contexts = link_deps,
            alwayslink = True,
        )
        linking_ctxs.append(link_ctx)

    # For now we don't compile anything.
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
    files_out.extend(out.library_to_link.pic_objects)
    cc_info = CcInfo(compilation_context = comp_ctx, linking_context = link)

    # TODO(allight): We don't want to include the generated files in sub-registry rules
    default_info = DefaultInfo(files = depset(
        # Ensure just building the registry does force all the passes to build.
        direct = files_out,
        transitive = [c[DefaultInfo].files for c in ctx.attr.passes],
    ))
    return [
        cc_info,
        default_info,
        registry_info_type(
            cc_library = cc_info,
            passes = passes,
            pipeline_binpb = pipeline_binpb,
            pass_infos = pass_infos,
            default_info = default_info,
            pipeline_src = ctx.file.pipeline,
        ),
        config_info_type(
            pipeline_binpb = pipeline_binpb,
            pass_infos = pass_infos,
        ),
    ]

def _xls_pass_registry_impl(ctx):
    return _xls_pass_registry_impl_helper(
        ctx = ctx,
        pass_info_type = XlsOptimizationPassInfo,
        registry_info_type = XlsOptimizationPassRegistryInfo,
        config_info_type = XlsOptimizationPassRegistryConfigInfo,
        proto_name = "xls.OptimizationPipelineProto",
        namespace = "xls::pass_registry",
    )

xls_pass_registry = rule(
    implementation = _xls_pass_registry_impl,
    doc = """Registry library that registers the given passes.

    NB Pass-registry dependencies do not add the compound passes etc to the namespace.

    TODO(allight): This would be a nice thing to do. Ensuring that overrides
    work reasonably would be required however.
    """,
    provides = [CcInfo, XlsOptimizationPassRegistryInfo, XlsOptimizationPassRegistryConfigInfo],
    fragments = ["cpp"],
    toolchains = use_cpp_toolchain(),
    attrs = dicts.add(
        {
            "passes": attr.label_list(
                doc = "List of pass infos, libraries or other pass_registry rules to register.",
                providers = [[XlsOptimizationPassRegistryInfo], [XlsOptimizationPassInfo], [CcInfo]],
            ),
            "pipeline": attr.label(
                doc = """Text proto OptimizationPipeline defining any compound passes and the default pipeline.

            At most one of this and pipeline_binpb may be present.""",
                allow_single_file = True,
                mandatory = False,
            ),
            "pipeline_binpb": attr.label(
                doc = """Binary proto OptimizationPipeline defining any compound passes and the default pipeline.

            At most one of this and pipeline may be present.""",
                allow_single_file = True,
                mandatory = False,
            ),
            "_register_pipeline_template": attr.label(
                default = Label("//xls/passes/tools:pipeline_registration.cc.tmpl"),
                allow_single_file = True,
            ),
            "_register_pipeline_deps": attr.label_list(
                default = [
                    Label("//xls/passes:optimization_pass_registry"),
                    Label("//xls/common:module_initializer"),
                    Label("@com_google_absl//absl/log:check"),
                ],
            ),
        },
        _proto_data_tool_attrs,
        _embed_data_attrs,
    ),
)

def _xls_codegen_pass_registry_impl(ctx):
    return _xls_pass_registry_impl_helper(
        ctx = ctx,
        pass_info_type = XlsCodegenPassInfo,
        registry_info_type = XlsCodegenPassRegistryInfo,
        config_info_type = XlsCodegenPassRegistryConfigInfo,
        proto_name = "xls.codegen.CodegenPipelineProto",
        namespace = "xls::codegen::pass_registry",
    )

xls_codegen_pass_registry = rule(
    implementation = _xls_codegen_pass_registry_impl,
    doc = """Registry library that registers the given codegen passes.""",
    provides = [CcInfo, XlsCodegenPassRegistryInfo, XlsCodegenPassRegistryConfigInfo],
    fragments = ["cpp"],
    toolchains = use_cpp_toolchain(),
    attrs = dicts.add(
        {
            "passes": attr.label_list(
                doc = "List of pass infos, libraries or other pass_registry rules to register.",
                providers = [[XlsCodegenPassRegistryInfo], [XlsCodegenPassInfo], [CcInfo]],
            ),
            "pipeline": attr.label(
                doc = """Text proto CodegenPipeline defining any compound passes and the default pipeline.

            At most one of this and pipeline_binpb may be present.""",
                allow_single_file = True,
                mandatory = False,
            ),
            "pipeline_binpb": attr.label(
                doc = """Binary proto CodegenPipeline defining any compound passes and the default pipeline.

            At most one of this and pipeline may be present.""",
                allow_single_file = True,
                mandatory = False,
            ),
            "_register_pipeline_template": attr.label(
                default = Label("//xls/codegen_v_1_5/tools:codegen_pipeline_registration.cc.tmpl"),
                allow_single_file = True,
            ),
            "_register_pipeline_deps": attr.label_list(
                default = [
                    Label("//xls/codegen_v_1_5:codegen_pass_registry"),
                    Label("//xls/common:module_initializer"),
                    Label("@com_google_absl//absl/log:check"),
                ],
            ),
        },
        _proto_data_tool_attrs,
        _embed_data_attrs,
    ),
)

def _xls_default_pass_registry(ctx):
    config = ctx.toolchains["//xls/common/toolchains:toolchain_type"].configuration
    return [
        config.pass_registry,
        config.pass_registry.cc_library,
        config.pass_registry.default_info,
        XlsOptimizationPassRegistryConfigInfo(
            pipeline_binpb = config.pass_registry.pipeline_binpb,
            pass_infos = config.pass_registry.pass_infos,
        ),
    ]

xls_default_pass_registry = rule(
    implementation = _xls_default_pass_registry,
    doc = """A pass registry with the default pipeline.""",
    provides = [XlsOptimizationPassRegistryInfo, XlsOptimizationPassRegistryConfigInfo, CcInfo],
    toolchains = ["//xls/common/toolchains:toolchain_type"],
)

def _xls_default_codegen_pass_registry(ctx):
    config = ctx.toolchains["//xls/common/toolchains:toolchain_type"].configuration
    return [
        config.codegen_pass_registry,
        config.codegen_pass_registry.cc_library,
        config.codegen_pass_registry.default_info,
        XlsCodegenPassRegistryConfigInfo(
            pipeline_binpb = config.codegen_pass_registry.pipeline_binpb,
            pass_infos = config.codegen_pass_registry.pass_infos,
        ),
    ]

xls_default_codegen_pass_registry = rule(
    implementation = _xls_default_codegen_pass_registry,
    doc = """A pass registry with the default codegen pipeline.""",
    provides = [CcInfo, XlsCodegenPassRegistryInfo, XlsCodegenPassRegistryConfigInfo],
    toolchains = ["//xls/common/toolchains:toolchain_type"],
)
