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
"""Helper functions for testing solvers on smt2 files."""

import subprocess
import time

SOLVER_COMMANDS = {
    "cvc4": "cvc4",
    "z3": "z3",
    "stp": "stp",
    "yices": "yices-smt2",
}


def get_solver_file_speed_ms(solver, f):
  """Returns a 10-run average (in ms) of the given solver on the given file.

  Args:
    solver: A string, the solver to be tested
    f: The smt2 file to test on

  Returns: the average time in ms.
  """
  avg = 0
  for _ in range(10):
    start = time.time()
    output = subprocess.check_output(
        [f"{SOLVER_COMMANDS[solver]}", f"{f.name}"]
    )
    if output != b"unsat\n":
      raise ValueError(f"{solver} messed up on {f.name}")
    end = time.time()
    avg += (1000 * (end - start)) / 10
  return avg


def get_solver_speeds_ms(solvers, files_list):
  """Returns the average speed of each solver on each of the files.

  Args:
    solvers: A list of strings, the solvers to test
    files_list: A list of smt2 files to test on

  Returns: a list of the average time per solver (in the order given).
  """
  solvers_milliseconds = []
  for solver in solvers:
    solver_milliseconds = []
    for f in files_list:
      solver_milliseconds.append(get_solver_file_speed_ms(solver, f))
    solvers_milliseconds.append(solver_milliseconds)
  return solvers_milliseconds
