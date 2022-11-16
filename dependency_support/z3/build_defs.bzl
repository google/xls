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
    "src/ackermannization/ackermannization_params.hpp",
    "src/ackermannization/ackermannize_bv_tactic_params.hpp",
    "src/ast/fpa/fpa2bv_rewriter_params.hpp",
    "src/ast/normal_forms/nnf_params.hpp",
    "src/ast/pattern/pattern_inference_params_helper.hpp",
    "src/ast/pp_params.hpp",
    "src/ast/rewriter/arith_rewriter_params.hpp",
    "src/ast/rewriter/array_rewriter_params.hpp",
    "src/ast/rewriter/bool_rewriter_params.hpp",
    "src/ast/rewriter/bv_rewriter_params.hpp",
    "src/ast/rewriter/fpa_rewriter_params.hpp",
    "src/ast/rewriter/poly_rewriter_params.hpp",
    "src/ast/rewriter/rewriter_params.hpp",
    "src/ast/rewriter/seq_rewriter_params.hpp",
    "src/math/polynomial/algebraic_params.hpp",
    "src/math/realclosure/rcf_params.hpp",
    "src/model/model_params.hpp",
    "src/model/model_evaluator_params.hpp",
    "src/muz/base/fp_params.hpp",
    "src/nlsat/nlsat_params.hpp",
    "src/opt/opt_params.hpp",
    "src/parsers/util/parser_params.hpp",
    "src/sat/sat_asymm_branch_params.hpp",
    "src/sat/sat_params.hpp",
    "src/sat/sat_scc_params.hpp",
    "src/sat/sat_simplifier_params.hpp",
    "src/smt/params/smt_params_helper.hpp",
    "src/solver/parallel_params.hpp",
    "src/solver/solver_params.hpp",
    "src/solver/combined_solver_params.hpp",
    "src/tactic/smtlogics/qfufbv_tactic_params.hpp",
    "src/tactic/sls/sls_params.hpp",
    "src/tactic/tactic_params.hpp",
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
        ] + native.glob(["src/**", "examples/**"]),
        tools = ["scripts/mk_make.py"],
        outs = MK_MAKE_SRCS + MK_MAKE_HDRS,
        # We can't use $(location) here, since the bundled script internally
        # makes assumptions about where files are located.
        cmd = "cd external/z3; " +
              "python scripts/mk_make.py; " +
              "cd ../..;" +
              copy_cmds,
    )

    for params_hdr in PARAMS_HDRS:
        src_file = params_hdr[0:-4] + ".pyg"
        native.genrule(
            name = "gen_" + params_hdr[0:-4],
            srcs = [src_file],
            tools = ["scripts/pyg2hpp.py"],
            outs = [params_hdr],
            cmd = "python $(location scripts/pyg2hpp.py) $< $$(dirname $@)",
        )

    for db_hdr in DB_HDRS:
        src = db_hdr[0:-1] + "smt2"
        native.genrule(
            name = "gen_" + db_hdr[0:-2],
            srcs = [src],
            tools = ["scripts/mk_pat_db.py"],
            outs = [db_hdr],
            cmd = "python $(location scripts/mk_pat_db.py) $< $@",
        )
