#
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

"""Performs ir-minimization using z3 on a pass pipeline while starting from a ir_fuzzer reproducer."""

from collections.abc import Sequence
import os
import subprocess
import tempfile

from absl import app
from absl import flags

from xls.common import runfiles

_TO_IR_BIN = runfiles.get_path("xls/fuzzer/ir_fuzzer/reproducer_to_ir_main")
_IR_EQUIV_BIN = runfiles.get_path("xls/dev_tools/check_ir_equivalence_main")
_IR_MIN_BIN = runfiles.get_path("xls/dev_tools/ir_minimizer_main")
_OPT_MAIN_BIN = runfiles.get_path("xls/tools/opt_main")

_PASS = flags.DEFINE_string(
    name="pass",
    default=None,
    required=True,
    help="The pass/pipeline which displays the incorrect optimization",
)
_FUZZTEST_ARGS = flags.DEFINE_integer(
    name="fuzztest_args",
    default=10,
    help=(
        "How many arguments the fuzzer was configured to create. Set to 0 if it"
        " was an IrFuzzPackage domain."
    ),
)
_REPRO = flags.DEFINE_string(
    name="repro",
    default=None,
    required=True,
    help=(
        "path to the fuzztest reproducer. May be a fuzztest repo target if"
        " reproducer_repo_files.cc can translate it."
    ),
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  with tempfile.TemporaryDirectory() as tmpdir:
    subprocess.run(
        [
            _TO_IR_BIN,
            _REPRO.value,
            f"--fuzztest_args={_FUZZTEST_ARGS.value}"
            if _FUZZTEST_ARGS.value > 0
            else "--nofuzztest_args",
            f"--ir_out={tmpdir}/fuzz.ir",
            f"--args_out={tmpdir}/args.binpb",
        ],
        check=True,
    )
    test_file = open(f"{tmpdir}/testit.sh", "wt")
    test_file.write("#!/bin/bash\n")
    test_file.write(f"""
if ! {_OPT_MAIN_BIN} '--passes={_PASS.value}' $1 --output_path={tmpdir}/fuzz.opt.ir --alsologtostderr;
then
  echo "Fails due to opt_main failure"
  exit 0
fi

{_IR_EQUIV_BIN} --mismatch_exit_code=0 --match_exit_code=1 $1 {tmpdir}/fuzz.opt.ir --alsologtostderr
exit $?""")
    test_file.close()
    os.chmod(f"{tmpdir}/testit.sh", 0o777)
    subprocess.run(
        [
            _IR_MIN_BIN,
            f"--test_executable={tmpdir}/testit.sh",
            "--can_remove_params",
            "--can_extract_segments",
            "--can_remove_sends",
            "--can_remove_receives",
            "--alsologtostderr",
            "--failed_attempt_limit=1000",
            f"{tmpdir}/fuzz.ir",
        ],
        check=True,
    )


if __name__ == "__main__":
  app.run(main)
