# Copyright 2021 The XLS Authors
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

"""
This module contains build macros for XLS.
"""

load(
    "//xls/build_rules:xls_codegen_rules.bzl",
    "append_xls_ir_verilog_generated_files",
    "get_xls_ir_verilog_generated_files",
    "validate_verilog_filename",
)
load(
    "//xls/build_rules:xls_config_rules.bzl",
    "enable_generated_file_wrapper",
)
load(
    "//xls/build_rules:xls_ir_rules.bzl",
    "append_xls_dslx_ir_generated_files",
    "append_xls_ir_opt_ir_generated_files",
    "get_xls_dslx_ir_generated_files",
    "get_xls_ir_opt_ir_generated_files",
)
load(
    "//xls/build_rules:xls_rules.bzl",
    "xls_dslx_verilog",
)

def xls_dslx_verilog_macro(
        name,
        dep,
        verilog_file,
        ir_conv_args = {},
        opt_ir_args = {},
        codegen_args = {},
        enable_generated_file = True,
        enable_presubmit_generated_file = False,
        **kwargs):
    """A macro wrapper for the 'xls_dslx_verilog' rule.

    The macro instantiates the 'xls_dslx_verilog' rule and
    'enable_generated_file_wrapper' function. The generated files of the rule
    are listed in the outs attribute of the rule.

    Args:
      name: The name of the rule.
      dep: The 'xls_dslx_module_library' target used for dependency. See 'dep'
        attribute from the 'xls_dslx_verilog' rule.
      ir_conv_args: IR conversion Arguments. See 'ir_conv_args' attribute from
        the 'xls_dslx_ir' rule.
      opt_ir_args: IR optimization Arguments. See 'opt_ir_args' attribute from
        the 'xls_ir_opt_ir' rule.
      codegen_args: Codegen Arguments. See 'codegen_args' attribute from the
        'xls_ir_verilog' rule.
      enable_generated_file: See 'enable_generated_file' from
        'enable_generated_file_wrapper' function.
      enable_presubmit_generated_file: See 'enable_presubmit_generated_file'
        from 'enable_generated_file_wrapper' function.
      **kwargs: Positional arguments. Named arguments.
    """

    # Type check input
    if type(name) != type(""):
        fail("Argument 'name' must be of string type.")
    if type(dep) != type(""):
        fail("Argument 'dep' must be of string type.")
    if type(verilog_file) != type(""):
        fail("Argument 'verilog_file' must be of string type.")
    if type(ir_conv_args) != type({}):
        fail("Argument 'ir_conv_args' must be of dictionary type.")
    if type(opt_ir_args) != type({}):
        fail("Argument 'opt_ir_args' must be of dictionary type.")
    if type(codegen_args) != type({}):
        fail("Argument 'codegen_args' must be of dictionary type.")
    if type(enable_generated_file) != type(True):
        fail("Argument 'enable_generated_file' must be of boolean type.")
    if type(enable_presubmit_generated_file) != type(True):
        fail("Argument 'enable_presubmit_generated_file' must be " +
             "of boolean type.")

    # Append output files to arguments.
    kwargs = append_xls_dslx_ir_generated_files(kwargs, name)
    kwargs = append_xls_ir_opt_ir_generated_files(kwargs, name)
    validate_verilog_filename(verilog_file)
    verilog_basename = verilog_file[:-2]
    kwargs = append_xls_ir_verilog_generated_files(
        kwargs,
        verilog_basename,
        codegen_args,
    )

    xls_dslx_verilog(
        name = name,
        dep = dep,
        verilog_file = verilog_file,
        ir_conv_args = ir_conv_args,
        opt_ir_args = opt_ir_args,
        codegen_args = codegen_args,
        outs = get_xls_dslx_ir_generated_files(kwargs) +
               get_xls_ir_opt_ir_generated_files(kwargs) +
               get_xls_ir_verilog_generated_files(kwargs, codegen_args) +
               [native.package_name() + "/" + verilog_file],
        **kwargs
    )
    enable_generated_file_wrapper(
        wrapped_target = name,
        enable_generated_file = enable_generated_file,
        enable_presubmit_generated_file = enable_presubmit_generated_file,
        **kwargs
    )

def xls_dslx_cpp_type_library(
        name,
        src):
    """Creates a cc_library target for transpiled DSLX types.

    This macros invokes the DSLX-to-C++ transpiler and compiles the result as
    a cc_library.

    Args:
      name: The name of the eventual cc_library.
      src: The DSLX file whose types to compile as C++.
    """
    native.genrule(
        name = name + "_generate_sources",
        srcs = [src],
        outs = [
            name + ".h",
            name + ".cc",
        ],
        tools = [
            "//xls/dslx:cpp_transpiler_main",
        ],
        cmd = "$(location //xls/dslx:cpp_transpiler_main) " +
              "--output_header_path=$(@D)/{}.h ".format(name) +
              "--output_source_path=$(@D)/{}.cc ".format(name) +
              "$(location {})".format(src),
    )

    native.cc_library(
        name = name,
        srcs = [":" + name + ".cc"],
        hdrs = [":" + name + ".h"],
        deps = [
            "@com_google_absl//absl/base:core_headers",
            "@com_google_absl//absl/status:status",
            "@com_google_absl//absl/status:statusor",
            "@com_google_absl//absl/types:span",
            "//xls/public:value",
        ],
    )

def xls_verify_checksum(name, src, out, sha256, visibility = None):
    """Verifies that 'src' has sha256sum exactly 'sha256'.

    Clones the src into 'out' in success, so users can take a dependence on 'out'
    and know that it is a successfully checksummed result according to the build
    process.

    This macro provides two facilities in the interest of paranoia:

    1. The genrule itself will fail / explode, which should cause the build to fail.
    2. A py_test is generated, so that if the user (accidentally) does not take a
       dependency on the checksummed output, a test is still present that will
       exhibit failure if a user runs a test wildcard on the containing directory.

    Args:
        name: Label given to this macro instantiation.
        src: Source file to checksum.
        out: Filename for the (checksum-verified) output.
        sha256: sha256 checksum value (hex string) to verify equal to the
          checksum of `src`.
        visibility: Optional visibility specifier for the resulting output file.
    """
    native.py_test(
        name = name + "_checksum_test",
        main = "verify_checksum.py",
        srcs = ["//xls/public:verify_checksum"],
        # Note: see definition of rootpath here:
        # https://docs.bazel.build/versions/2.0.0/be/make-variables.html#predefined_label_variables
        args = ["$(rootpath %s)" % src, sha256],
        data = [src],
    )
    native.genrule(
        name = name,
        srcs = [src],
        outs = [out],
        visibility = visibility,
        # Capture the sha256sum output, only need the first token (the hash
        # itself), note what we got in a file for convenient debugging, then
        # compare to what the user specified and explode if it's not what they
        # wanted. If it is, proceed to write src to out.
        cmd = "SHA256_OUT=$$(sha256sum $<); GOT_HASH=$${SHA256_OUT%%%% *}; echo $${GOT_HASH} > $@.sha256; [[ \"$${GOT_HASH}\" == \"%s\" ]] || exit 1; cat $< > $@" % sha256,
        message = "Validating sha256 checksum for %s" % src,
    )
