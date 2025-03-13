// RUN: xls_opt %s | FileCheck %s

// Test that codegen attributes parse and print correctly
module attributes {
  xls.scheduling_options = #xls.scheduling_options<clock_period_ps = 1000, pipeline_stages = 5, delay_model = "unit", clock_margin_percent = 20, worst_case_throughput = 2, multi_proc = true>,
  xls.opt_options = #xls.opt_options<opt_level = 2, convert_array_index_to_select = 10, use_context_narrowing_analysis = true, optimize_for_best_case_throughput = true, enable_resource_sharing = true, force_resource_sharing = false, pass_pipeline = "">,
  xls.codegen_options = #xls.codegen_options<use_system_verilog = false, array_index_bounds_checking = false, gate_recvs = true>
} {
  // CHECK: xls.scheduling_options = #xls.scheduling_options<clock_period_ps = 1000, pipeline_stages = 5, delay_model = "unit", clock_margin_percent = 20, worst_case_throughput = 2, multi_proc = true>
  // CHECK: xls.opt_options = #xls.opt_options<opt_level = 2, convert_array_index_to_select = 10, use_context_narrowing_analysis = true, optimize_for_best_case_throughput = true, enable_resource_sharing = true, force_resource_sharing = false, pass_pipeline = "">
  // CHECK: xls.codegen_options = #xls.codegen_options<use_system_verilog = false, array_index_bounds_checking = false, gate_recvs = true>
  
  func.func @dummy() {
    return
  }
}
