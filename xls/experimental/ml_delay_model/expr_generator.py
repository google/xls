#
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
"""Generates a sequence of operations in Reverse Polish Notation.

Runs the verilog through yosys and nexpnr to get the delay estimate.
"""

import argparse
import csv
import enum
import multiprocessing as mp
import os
import random
import re
import subprocess
import tempfile
import time
from typing import Tuple, Type

import numpy as np

NUMOPS = 8
MAXLEN = 8
NUM_PROCESSES = 16


class Op(enum.Enum):
  NOP = 0
  PARAM1 = 1
  PARAM2 = 2
  ADD2 = 3
  NEG = 4
  XOR2 = 5
  MUL2 = 6
  AND2 = 7
  OR2 = 8


consumed_stack_slots = {
    Op.NOP: 0,
    Op.PARAM1: 0,
    Op.PARAM2: 0,
    Op.ADD2: 2,
    Op.NEG: 1,
    Op.XOR2: 2,
    Op.MUL2: 2,
    Op.AND2: 2,
    Op.OR2: 2,
}

to_str = {
    Op.PARAM1: 'x1',
    Op.PARAM2: 'x2',
    Op.ADD2: '+',
    Op.NEG: '~',
    Op.XOR2: '^',
    Op.MUL2: '*',
    Op.AND2: '&',
    Op.OR2: '|',
}

VERILOG_TEMPLATE = """
module my_module(
        input wire [3:0] x1,
        input wire [3:0] x2,
        output [3:0] out
);
        assign out = {};
endmodule
"""

binary_ops = [Op.ADD2, Op.XOR2, Op.MUL2, Op.AND2, Op.OR2]


def gen() -> Tuple[Type[Op], Type[str]]:
  """Returns random sequence of ops in RPN & corresponding Verilog."""
  oplist = list(Op)
  while True:
    ops = [Op.PARAM1, Op.PARAM2]
    stack = ['x1', 'x2']
    # First pick a length limit, using exponential dropoff probability
    length = np.random.choice(
        np.arange(1, MAXLEN + 1),
        p=[(1 << i) / ((1 << MAXLEN) - 1) for i in range(MAXLEN)],
    )
    while len(ops) < length:
      op = random.choice([op for op in oplist[3:]])
      # Append random operands
      while consumed_stack_slots[op] > len(stack):
        new_op = random.choice([op for op in oplist[1:3]])
        stack.append(to_str[new_op])
        ops.append(new_op)
      ops.append(op)
      if op in binary_ops:
        arg1 = stack.pop()
        arg2 = stack.pop()
        stack.append('({}{}{})'.format(arg1, to_str[op], arg2))
      elif op == Op.NEG:
        arg = stack.pop()
        stack.append('~{}'.format(arg))
    # Only use if all operands are used up.
    if len(stack) == 1:
      while len(ops) < MAXLEN:
        ops.append(Op.NOP)
      if len(ops) > MAXLEN:
        continue
      return ops, stack[0]


def parse_log(filename: str) -> str:
  """Extracts and returns the delay from the given log file path."""
  with open(filename, 'r') as log_file:
    info = log_file.read()
    delay_statement = 'Max delay <async> -> <async>: '
    locs = [m.start() for m in re.finditer(delay_statement, info)]
    # If there is no delay statement, we know it has been optimized
    # to a constant.
    if not locs:
      return '0.0'
    # Want to extract the final delay
    idx = locs[-1] + len(delay_statement)
    end = idx
    while info[end] != ' ':
      end += 1
    return info[idx:end]


def yosys_and_nextpnr(expr: str) -> float:
  """Runs Yosys and nextpnr tools to get delay estimate."""
  with tempfile.TemporaryDirectory() as tempdir:
    with open('{}/sample.v'.format(tempdir), 'w+') as verilog_file, open(
        '{}/sample.json'.format(tempdir), 'w+'
    ) as json_file, open('{}/sample.log'.format(tempdir), 'w+') as log_file:
      verilog_file.write(VERILOG_TEMPLATE.format(expr))
    subprocess.run(
        [
            '/usr/local/google/home/brjiang/Documents/yosys/yosys',
            '-p',
            'read_verilog {}; synth_ecp5 -top my_module -json {}'.format(
                verilog_file.name, json_file.name
            ),
        ],
        stdout=subprocess.DEVNULL,
        check=True,
    )
    subprocess.run(
        [
            'nextpnr-ecp5',
            '--json',
            json_file.name,
            '--package',
            'CABGA381',
            '--log',
            log_file.name,
        ],
        stderr=subprocess.DEVNULL,
        check=True,
    )
    delay = parse_log(log_file.name)
  return delay


def gen_csv(num_samples: int, name: str):
  with open(name, 'w+') as f:
    writer = csv.writer(f, delimiter=',')
    for _ in range(num_samples):
      ops, expr = gen()
      codevec = [str(op.value) for op in ops]
      codevec.append(str(yosys_and_nextpnr(expr)))
      writer.writerow(codevec)


def main(num_samples):
  start = time.time()
  processes = []
  for i in range(NUM_PROCESSES):
    processes.append(
        mp.Process(
            target=gen_csv,
            args=[
                num_samples // NUM_PROCESSES,
                './data/data_{}_{}_{}.csv'.format(NUMOPS, MAXLEN, i),
            ],
        )
    )
    processes[i].start()
  for i in range(NUM_PROCESSES):
    processes[i].join()
  end = time.time()
  print('Time elapsed: {} s'.format(end - start))
  # Append individual files back together
  with open('./data/data_{}_{}.csv'.format(NUMOPS, MAXLEN), 'w+') as outfile:
    for i in range(NUM_PROCESSES):
      filename = './data/data_{}_{}_{}.csv'.format(NUMOPS, MAXLEN, i)
      with open(filename, 'r') as f:
        for line in f:
          outfile.write(line)
      os.remove(filename)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('num_samples', type=int)
  args = parser.parse_args()
  main(args.num_samples)
