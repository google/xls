"""
This file contains a function that takes in a csv file and plots the data in within it. 
This file is meant to be used in conjunction with solvers_op_comparison_bits_list.py and
solvers_op_comparison_nests_list.py. Those two files produce the data as csv files, and
this file reads that data and plots it. 
"""
import csv
from matplotlib import pyplot as plt
from xls.common.gfile import open as gopen

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("fname", None, "csv file with data to plot")
flags.mark_flag_as_required("fname")

def plot_csv_data(fname):
  """
  Plot the contents of the csv file fname. 

  Given a csv file, takes the first row as the values for the x-axis, and each of the
  following rows as data for a solver. Assumes the first element in each row is the 
  name of the solver. 

  Args:
  fname: The name of the file containing the data to plot. 
  """
  with gopen(fname, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    line_count = 0
    for row in csv_reader:
      if line_count == 0:
        x_axis = [int(elm) for elm in row[1:]]
        if row[0] == "bits_list":
          xlabel = "Bit Count"
        elif row[0] == "nests_list":
          xlabel = "Nested Ops"
      else:
        data = [float(elm) for elm in row[1:]]
        label = row[0]
        plt.scatter(x_axis, data, label=label)
        print(f"row[1:] = {row[1:]}, label = {row[0]}")
      line_count += 1
    plt.xscale("log", basex=2)
    plt.xlabel(xlabel)
    plt.ylabel("Solve Time (ms)")
    plt.legend()
    plt.show()

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  plot_csv_data(FLAGS.fname)

if __name__ == '__main__':
  app.run(main)
