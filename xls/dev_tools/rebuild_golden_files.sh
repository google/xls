#!/bin/sh -ex

# Rebuilds the golden files of all tests using
# `xls/common/golden_files.h` infrastructure.
#
# Run with no arguments will rebuild golden files for all such targets. Can also
# be called with a list of test targets to rebuild.

TARGETS=(
"//xls/codegen:block_generator_test"
"//xls/codegen:combinational_generator_test"
"//xls/codegen:finite_state_machine_test"
"//xls/codegen:module_builder_test"
"//xls/codegen:pipeline_generator_test"
"//xls/contrib/xlscc/unit_tests:translator_verilog_test"
"//xls/dslx/cpp_transpiler:cpp_transpiler_test"
"//xls/dslx/ir_convert:function_converter_test"
"//xls/dslx/ir_convert:ir_converter_test"
"//xls/dslx/type_system:type_info_to_proto_test"
"//xls/fdo:synthesizer_test"
"//xls/flows:ir_wrapper_test"
"//xls/ir:block_test"
"//xls/ir:ir_parser_round_trip_test"
"//xls/simulation:module_testbench_test"
"//xls/simulation:verilog_test_base_test"
"//xls/tools:codegen_main_test"
"//xls/contrib/ice40:wrap_io_test"
"//xls/visualization/ir_viz:ir_to_proto_test"
)

if [[ "$@" ]]
then
  TARGETS=($@)
fi

if [[ ! -f "$(pwd)/WORKSPACE" ]]
then
  echo "Must be run from root repo directory"
  exit 1
fi

# Some dependencies do not build properly with --spawn_strategyy=standalone so
# build the targets normally first.
bazel build -c opt ${TARGETS[@]}

bazel test -c opt \
  --test_strategy=standalone \
  --spawn_strategy=standalone \
  ${TARGETS[@]} \
  --test_arg=--test_update_golden_files \
  --test_arg=--xls_source_dir="$(pwd)"/xls/ \
  --test_arg=--alsologtostderr \
  --nocache_test_results \
  --test_output=errors
