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

"""Macros for generating Z3 input sources/headers."""

MK_MAKE_SRCS = [
    "src/api/api_commands.cpp",
    "src/api/api_log_macros.cpp",
    "src/api/dll/gparams_register_modules.cpp",
    "src/api/dll/install_tactic.cpp",
    "src/api/dll/mem_initializer.cpp",
    "src/util/z3_version.h",
]

MK_MAKE_HDRS = [
    "src/api/api_log_macros.h",
]

PARAMS_HDRS = [
    "src/model/model_evaluator_params.hpp",
    "src/model/model_params.hpp",
    "src/ast/normal_forms/nnf_params.hpp",
    "src/ast/pp_params.hpp",
    "src/ackermannization/ackermannize_bv_tactic_params.hpp",
    "src/ackermannization/ackermannization_params.hpp",
    "src/math/realclosure/rcf_params.hpp",
    "src/math/polynomial/algebraic_params.hpp",
    "src/tactic/smtlogics/qfufbv_tactic_params.hpp",
    "src/params/poly_rewriter_params.hpp",
    "src/params/array_rewriter_params.hpp",
    "src/params/seq_rewriter_params.hpp",
    "src/params/fpa_rewriter_params.hpp",
    "src/params/arith_rewriter_params.hpp",
    "src/params/rewriter_params.hpp",
    "src/params/bv_rewriter_params.hpp",
    "src/params/tactic_params.hpp",
    "src/params/pattern_inference_params_helper.hpp",
    "src/params/solver_params.hpp",
    "src/params/fpa2bv_rewriter_params.hpp",
    "src/params/bool_rewriter_params.hpp",
    "src/params/sat_params.hpp",
    "src/params/sls_params.hpp",
    "src/solver/combined_solver_params.hpp",
    "src/solver/parallel_params.hpp",
    "src/opt/opt_params.hpp",
    "src/parsers/util/parser_params.hpp",
    "src/muz/base/fp_params.hpp",
    "src/sat/sat_asymm_branch_params.hpp",
    "src/sat/sat_simplifier_params.hpp",
    "src/sat/sat_scc_params.hpp",
    "src/nlsat/nlsat_params.hpp",
    "src/smt/params/smt_params_helper.hpp",
]

DB_HDRS = [
    "src/ast/pattern/database.h",
]

GEN_SRCS = MK_MAKE_SRCS

GEN_HDRS = MK_MAKE_HDRS + PARAMS_HDRS + DB_HDRS

def gen_srcs():
    """Runs provided Python scripts for source generation."""
    copy_cmds = []
    for out in MK_MAKE_SRCS + MK_MAKE_HDRS:
        copy_cmds.append("cp external/z3/" + out + " $(@D)/$$(dirname " + out + ")")
    copy_cmds = ";\n".join(copy_cmds) + ";\n"

    native.genrule(
        name = "gen_srcs",
        srcs = [
            "LICENSE.txt",
            "scripts/mk_make.py",
        ] + native.glob(["src/**", "examples/**"], exclude = GEN_HDRS + GEN_SRCS),
        tools = ["scripts/mk_make.py"],
        outs = MK_MAKE_SRCS + MK_MAKE_HDRS,
        # We can't use $(location) here, since the bundled script internally
        # makes assumptions about where files are located.
        cmd = "python3=`realpath $(PYTHON3)`; " +
              "cd external/z3; " +
              "$$python3 scripts/mk_make.py; " +
              "cd ../..;" +
              copy_cmds,
        toolchains = ["@rules_python//python:current_py_toolchain"],
    )

    for params_hdr in PARAMS_HDRS:
        src_file = params_hdr[0:-4] + ".pyg"
        native.genrule(
            name = "gen_" + params_hdr[0:-4],
            srcs = [src_file],
            tools = ["scripts/pyg2hpp.py"],
            outs = [params_hdr],
            cmd = "$(PYTHON3) $(location scripts/pyg2hpp.py) $< $$(dirname $@)",
            toolchains = ["@rules_python//python:current_py_toolchain"],
        )

    for db_hdr in DB_HDRS:
        src = db_hdr[0:-1] + "smt2"
        native.genrule(
            name = "gen_" + db_hdr[0:-2],
            srcs = [src],
            tools = ["scripts/mk_pat_db.py"],
            outs = [db_hdr],
            cmd = "$(PYTHON3) $(location scripts/mk_pat_db.py) $< $@",
            toolchains = ["@rules_python//python:current_py_toolchain"],
        )
