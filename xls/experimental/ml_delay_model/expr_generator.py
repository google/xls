"""
Generates a sequence of operations in Reverse Polish Notation,
and runs the verilog through yosys and nexpnr to get the delay
estimate.
"""

from enum import Enum
import random
from typing import Tuple, Type
import tempfile
import subprocess
import numpy as np
import re
import csv
from multiprocessing import Process
import os
import time
import argparse


NUMOPS = 8
MAXLEN = 8
NUM_PROCESSES = 16
BITWIDTH = 4
XLS_DIRECTORY = "/usr/local/google/home/brjiang/Documents/xls-1"

class Op(Enum):
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
  Op.OR2: 2
}

to_str = {
  Op.PARAM1: "x1",
  Op.PARAM2: "x2",
  Op.ADD2: "+",
  Op.NEG: "~",
  Op.XOR2: "^",
  Op.MUL2: "*",
  Op.AND2: "&",
  Op.OR2: "|"
}

"""
get_ir_fn = {
  Op.ADD2: fb.add_or,
  Op.NEG: fb.add_not,
  Op.XOR2: fb.add_xor,
  Op.MUL2: fb.add_umul,
  Op.AND2: fb.add_and,
  Op.OR2: fb.add_or
}
"""

VERILOG_TEMPLATE = """
module my_module(
        input wire [3:0] x1,
        input wire [3:0] x2,
        output [3:0] out
);
        assign out = {};
endmodule
"""

DSLX_TEMPLATE = """
fn func(x1: u4, x2: u4) -> u4 {{
  {}
}}
"""

binary_ops = [Op.ADD2, Op.XOR2, Op.MUL2, Op.AND2, Op.OR2]


def gen() -> Tuple[Type[Op], Type[str]]:
  """
  Generates a random sequence of operations in Reverse Polish Notation,
  corresponding verilog, and XLS IR.
  """
  oplist = list(Op)
  while True:
    ops = [Op.PARAM1, Op.PARAM2]
    stack = ["x1", "x2"]

    # First pick a length limit, using exponential dropoff probability
    length = np.random.choice(np.arange(1, MAXLEN+1),
                              p=[(1 << i) / ((1 << MAXLEN) - 1) \
                                 for i in range(MAXLEN)])

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
        stack.append("({}{}{})".format(arg1, to_str[op], arg2))
      elif op == Op.NEG:
        arg = stack.pop()
        stack.append("~{}".format(arg))
    # Only use if all operands are used up.
    if len(stack) == 1:
      while len(ops) < MAXLEN:
        ops.append(Op.NOP)
      if len(ops) > MAXLEN:
        continue
      return ops, stack[0]


def parse_log(filename: str) -> str:
  with open(filename, "r") as log_file:
    info = log_file.read()
    delay_statement = 'Max delay <async> -> <async>: '
    locs = [m.start() for m in re.finditer(delay_statement, info)]
    # If there is no delay statement, we know it has been optimized
    # to a constant.
    if len(locs) == 0:
      return "0.0"
    # Want to extract the final delay
    idx = locs[-1] + len(delay_statement)
    end = idx
    while info[end] != " ":
      end += 1
    return info[idx:end]


def yosys_and_nextpnr(expr: str) -> float:
  """
  Runs Yosys and nextpnr tools to get delay estimate.
  """
  with tempfile.TemporaryDirectory() as tempdir:
    with open("{}/sample.v".format(tempdir), "w+") as verilog_file, \
        open("{}/sample.json".format(tempdir), "w+") as json_file, \
        open("{}/sample.log".format(tempdir), "w+") as log_file:
      verilog_file.write(VERILOG_TEMPLATE.format(expr))
    subprocess.run(["/usr/local/google/home/brjiang/Documents/yosys/yosys","-p",
                    "read_verilog {}; synth_ecp5 -top my_module -json {}"
                      .format(verilog_file.name, json_file.name)],
                   stdout=subprocess.DEVNULL)
    subprocess.run(["nextpnr-ecp5", "--json", json_file.name,
                    "--package", "CABGA381",
                    "--log", log_file.name],
                    stderr=subprocess.DEVNULL)
    delay = parse_log(log_file.name)
  return delay


def parse_benchmark_output(filename: str) -> str:
  with open(filename, "r") as out_file:
    info = out_file.read()
    delay_statement = "Return value delay: "
    idx = info.find(delay_statement)
    if idx == -1:
      print("No delay")
    else:
      idx += len(delay_statement)
      # Delay is actually returned in ps, so we need to convert it
      end = idx
      while info[end].isdigit():
        end += 1
      return float(info[idx:end]) / 1000

def get_curr_estimate(expr: str) -> float:
  """
  Use current XLS model to get delay estimate. Must first build
  corresponding XLS targets first.
  """
  with tempfile.TemporaryDirectory() as tempdir:
    with open("{}/sample.x".format(tempdir), "w+") as dslx_file, \
        open("{}/sample.out".format(tempdir), "w+") as out_file:
      # Bitwise negation is ! in DSLX but ~ in Verilog
      dslx_file.write(DSLX_TEMPLATE.format(expr).replace('~', '!'))
      #print(dslx_file.read())
    with open("{}/sample.ir".format(tempdir), "w+") as ir_file:
      s = subprocess.run(["{}/bazel-bin/xls/dslx/ir_converter_main".format(XLS_DIRECTORY),
                      dslx_file.name], capture_output=True)
      print(s.stdout.decode("utf-8"), file=ir_file)
    with open("{}/sample.out".format(tempdir), "w+") as out_file:
      s = subprocess.run(["{}/bazel-bin/xls/tools/benchmark_main".format(XLS_DIRECTORY),
                          ir_file.name], capture_output=True)
      print(s.stdout.decode("utf-8"), file=out_file)
    return parse_benchmark_output(out_file.name)


def gen_csv(num_samples: int, name: str):
  with open(name, "w+") as f:
    writer = csv.writer(f, delimiter=',')
    for _ in range(num_samples):
      ops, expr = gen()
      codevec = [str(op.value) for op in ops]
      codevec.append(str(yosys_and_nextpnr(expr)))
      codevec.append(get_curr_estimate(expr))
      writer.writerow(codevec)


bitwidths = [2**i for i in range(11)]
def collect_data():
  expr = ""
  for op in list(Op)[3:]:
    if consumed_stack_slots[op] == 1:
      expr = "{}x1".format(to_str[op])
    elif consumed_stack_slots[op] == 2:
      expr = "x1{}x2".format(to_str[op])
    delay = yosys_and_nextpnr(expr)
    print(to_str[op], delay)


def main(num_samples):
  start = time.time()
  processes = []
  for i in range(NUM_PROCESSES):
    processes.append(Process(target=gen_csv,
                             args=[num_samples//NUM_PROCESSES,
                                   "./data/data_{}_{}_{}.csv".format(
                                       NUMOPS, MAXLEN, i)]))
    processes[i].start()
  for i in range(NUM_PROCESSES):
    processes[i].join()
  end = time.time()
  print('Time elapsed: {} s'.format(end-start))
  # Append individual files back together
  with open("./data/data_{}_{}.csv".format(NUMOPS,MAXLEN), "w+") as outfile:
    for i in range(NUM_PROCESSES):
      filename = "./data/data_{}_{}_{}.csv".format(NUMOPS,MAXLEN, i)
      with open(filename, "r") as f:
        for line in f:
          outfile.write(line)
      os.remove(filename)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('num_samples', type=int)
  args = parser.parse_args()
  main(args.num_samples)
