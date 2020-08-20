"""
This file contains helper functions for testing solvers on smt2 files. 

get_solver_file_speed_ms takes in a solver and a file. It tests the solver 10 times
on the file (raising an error if the solver doesn't produce the expected result), 
and reutrn the average time (in milliseconds) the 10 tests took.

get_solver_speeds_ms takes in a list of solvers and a list of files. It tests each 
solver on each of the files using get_solver_file_speed_ms, and returns a list of 
lists containing the average speed for each of the files, for each of the solvers.
"""

import subprocess
import time

SOLVER_COMMANDS = {"cvc4": "cvc4",
                   "z3": "z3",
                   "stp": "stp",
                   "yices": "yices-smt2"}

def get_solver_file_speed_ms(solver, f):
  """
  Return a 10-run average (in milliseconds) of the given solver on the given file. 

  Args:
  solver: A string, the solver to be tested
  f: The smt2 file to test on
  """
  avg = 0
  for _ in range(10):
    start = time.time()
    output = subprocess.check_output(
        [f"{SOLVER_COMMANDS[solver]}", 
         f"{f.name}"])
    if output != b"unsat\n":
      raise ValueError(f"{solver} messed up on {f.name}")
    end = time.time()
    avg += (1000 * (end - start)) / 10 
  return avg

def get_solver_speeds_ms(solvers, files_list):
  """
  Return the average speed of each solver on each of the files. 

  Args:
  solvers: A list of strings, the solvers to test
  files_list: A list of smt2 files to test on 
  """
  solvers_milliseconds = []
  for solver in solvers:
    solver_milliseconds = []
    for f in files_list:
      solver_milliseconds.append(get_solver_file_speed_ms(solver, f))
    solvers_milliseconds.append(solver_milliseconds)
  return solvers_milliseconds

