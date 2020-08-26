# Copyright 2020 Google LLC
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
This file takes in 5 flags:

--op: a string representing the operation (can be 'add', 'mul', or 'shl')

--nests: an integer representing the number of nested operations

--bits_list: a list of integers representing the bitvector length for each proof

--solvers: a list of strings representing each of the solvers to be tested

--fname: a file name to store the data (if fname already exists, data is appended
         to the end of the file)

This script creates smt2 files for each proof, runs each of the solvers on these
smt2 files, and stores the speed of each solver on each smt2 file in a csv file.
This data can then be plotted using plot_csv_solver_speed_data.py.
"""
import os
import sys
import csv
from matplotlib import pyplot as plt
from n_bit_nested_add_generator import n_bit_nested_add_existing_file
from n_bit_nested_mul_generator import n_bit_nested_mul_existing_file
from n_bit_nested_shift_generator import n_bit_nested_shift_existing_file
from solvers_op_comparison_functions import *
from xls.common.gfile import open as gopen
from flags_checks import *

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("op", None, "Operation for the smt2 files (add, mul, shl)")
flags.DEFINE_integer("nests", None, "Integer for the number of nested operations.")
flags.DEFINE_list("bits_list", None, "List of n values for each n-bit multiplication proof.")
flags.register_validator("bits_list",
                         list_contains_only_integers,
                         message="--bits_list must contain only integers.")
flags.DEFINE_list("solvers", None, "List of solvers to test.")
flags.DEFINE_string("fname", None, "Name for the file to store the data.")

flags.mark_flag_as_required("op")
flags.mark_flag_as_required("nests")
flags.mark_flag_as_required("bits_list")
flags.mark_flag_as_required("fname")
flags.mark_flag_as_required("solvers")

def create_and_get_smt_files_bits_list(op, nests_val, bits_list):
  """
  Create smt2 files for the necessary proof and return them in a list.

  Given an operation, the number of nests, and a list of bits, create smt2
  files for each proof, and return them in a list.

  Args:
  op: A string, the operation to test ('add', 'mul', or 'shl')
  nests_val: An integer, the number of nested operations
  bits_list: A list of integers, the input bitvector length for each proof
  """
  if op not in ["add", "mul", "shl"]:
    raise ValueError("op argument is not a valid operation")
  files = []
  for bits in bits_list:
    with gopen(f"{op}{nests_val}_2x{bits}.smt2", "w+") as f:
      files.append(f)
      if op == "add":
        n_bit_nested_add_existing_file(bits, nests_val, f)
      elif op == "mul":
        n_bit_nested_mul_existing_file(bits, nests_val, f)
      elif op == "shl":
        n_bit_nested_shift_existing_file(bits, nests_val, f)
  return files

def csv_solvers_speeds_bits_list(op, nests_val, bits_list, solvers, fname):
  """
  Create smt2 files for each proof, test the solvers, and write the data to a csv file.

  Create smt2 files for each of the bitvector lengths in bits_list with the
  given operation and number of nested operations, get the average speeds of
  each of the solvers on these proofs, and store the data in a csv file with fname.

  Args:
  op: A string, the operation to test
  nests_val: An integer, the number of nested operations
  bits_list: A list of integers, the input bitvector length for each proof
  solvers: A list of strings, the solvers to test
  fname: The name of the file to store the data in
  """
  files = create_and_get_smt_files_bits_list(op, nests_val, bits_list)
  solvers_milliseconds = get_solver_speeds_ms(solvers, files)
  write_row = False if os.path.isfile(fname) else True
  with gopen(fname, "a") as f:
    wr = csv.writer(f, delimiter=",")
    if write_row:
      wr.writerow(["bits_list"] + bits_list)
    for i in range(len(solvers)):
      wr.writerow([solvers[i]] + solvers_milliseconds[i])

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  else:
    op = FLAGS.op
    nests_val = FLAGS.nests
    bits_list = [int(elm) for elm in FLAGS.bits_list]
    solvers = FLAGS.solvers
    fname = FLAGS.fname
    csv_solvers_speeds_bits_list(op, nests_val, bits_list, solvers, fname)

if __name__ == '__main__':
  app.run(main)

