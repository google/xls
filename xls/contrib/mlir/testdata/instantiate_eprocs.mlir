// RUN: xls/contrib/mlir/xls_opt -instantiate-eprocs -symbol-dce -split-input-file %s 2>&1 | FileCheck %s

// CHECK-LABEL: xls.chan @InstantiateTwice : i32
// CHECK: xls.eproc @p_1
// CHECK-SAME: min_pipeline_stages = 3
// CHECK: xls.blocking_receive %0, @InstantiateTwice
// CHECK: xls.eproc @p_0
// CHECK-SAME: min_pipeline_stages = 3
// CHECK: xls.blocking_receive %0, @InstantiateTwice2
xls.chan @InstantiateTwice : i32
xls.chan @InstantiateTwice2 : i32

xls.chan @Local : i32
xls.eproc @p(%arg0: i32) zeroinitializer discardable attributes {min_pipeline_stages = 3 : i64} {
  %0 = xls.after_all  : !xls.token
  %tkn_out, %result = xls.blocking_receive %0, @Local : i32
  xls.yield %arg0 : i32
}

xls.instantiate_eproc @p (@Local as @InstantiateTwice)
xls.instantiate_eproc @p (@Local as @InstantiateTwice2)

// -----

// CHECK-LABEL:  xls.chan @IntegrationTestLabel : i32
// CHECK-NEXT:  xls.chan @resp : i32
// CHECK-NEXT:  xls.chan @rom1_req : i32
// CHECK-NEXT:  xls.chan @rom1_resp : i32
// CHECK-NEXT:  xls.eproc @rom_0_4(%arg0: i32) zeroinitializer {
// CHECK-NEXT:    %0 = "xls.constant_scalar"() <{value = 1 : i32}> : () -> i32
// CHECK-NEXT:    %1 = xls.after_all  : !xls.token
// CHECK-NEXT:    %tkn_out, %result = xls.blocking_receive %1, @rom1_req : i32
// CHECK-NEXT:    %2 = xls.send %tkn_out, %0, @rom1_resp : i32
// CHECK-NEXT:    xls.yield %arg0 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  xls.chan @rom2_req : i32
// CHECK-NEXT:  xls.chan @rom2_resp : i32
// CHECK-NEXT:  xls.eproc @rom_0_3(%arg0: i32) zeroinitializer {
// CHECK-NEXT:    %0 = "xls.constant_scalar"() <{value = 1 : i32}> : () -> i32
// CHECK-NEXT:    %1 = xls.after_all  : !xls.token
// CHECK-NEXT:    %tkn_out, %result = xls.blocking_receive %1, @rom2_req : i32
// CHECK-NEXT:    %2 = xls.send %tkn_out, %0, @rom2_resp : i32
// CHECK-NEXT:    xls.yield %arg0 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  xls.eproc @proxy_0_2(%arg0: i32) zeroinitializer {
// CHECK-NEXT:    %0 = xls.after_all  : !xls.token
// CHECK-NEXT:    %tkn_out, %result = xls.blocking_receive %0, @IntegrationTestLabel : i32
// CHECK-NEXT:    %1 = xls.send %tkn_out, %result, @rom1_req : i32
// CHECK-NEXT:    %tkn_out_0, %result_1 = xls.blocking_receive %1, @rom1_resp : i32
// CHECK-NEXT:    %2 = xls.send %tkn_out_0, %result_1, @resp : i32
// CHECK-NEXT:    xls.yield %arg0 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  xls.eproc @fetch_0_1(%arg0: i32) zeroinitializer {
// CHECK-NEXT:    %0 = xls.after_all  : !xls.token
// CHECK-NEXT:    %1 = xls.send %0, %arg0, @IntegrationTestLabel : i32
// CHECK-NEXT:    %tkn_out, %result = xls.blocking_receive %1, @resp : i32
// CHECK-NEXT:    xls.yield %result : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  xls.chan @boundary1 {send_supported = false} : i32
// CHECK-NEXT:  xls.chan @boundary2 {recv_supported = false} : i32
// CHECK-NEXT:  xls.eproc @rom_0_0(%arg0: i32) zeroinitializer {
// CHECK-NEXT:    %0 = "xls.constant_scalar"() <{value = 1 : i32}> : () -> i32
// CHECK-NEXT:    %1 = xls.after_all  : !xls.token
// CHECK-NEXT:    %tkn_out, %result = xls.blocking_receive %1, @boundary1 : i32
// CHECK-NEXT:    %2 = xls.send %tkn_out, %0, @boundary2 : i32
// CHECK-NEXT:    xls.yield %arg0 : i32
// CHECK-NEXT:  }

xls.chan @IntegrationTestLabel : i32
xls.chan @resp : i32
xls.chan @rom1_req : i32
xls.chan @rom1_resp : i32
xls.eproc @rom_0(%arg0: i32) zeroinitializer discardable {
  %0 = xls.after_all  : !xls.token
  %tkn_out, %result = xls.blocking_receive %0, @rom_arg0 : i32
  %1 = "xls.constant_scalar"() <{value = 1 : i32}> : () -> i32
  %2 = xls.send %tkn_out, %1, @rom_arg1 : i32
  xls.yield %arg0 : i32
}
xls.chan @rom_arg0 : i32
xls.chan @rom_arg1 : i32
xls.instantiate_eproc @rom_0 (@rom_arg0 as @rom1_req, @rom_arg1 as @rom1_resp)
xls.chan @rom2_req : i32
xls.chan @rom2_resp : i32
xls.instantiate_eproc @rom_0 (@rom_arg0 as @rom2_req, @rom_arg1 as @rom2_resp)
xls.eproc @proxy_0(%arg0: i32) zeroinitializer discardable {
  %0 = xls.after_all  : !xls.token
  %tkn_out, %result = xls.blocking_receive %0, @proxy_arg0 : i32
  %1 = xls.send %tkn_out, %result, @proxy_arg2 : i32
  %tkn_out_0, %result_1 = xls.blocking_receive %1, @proxy_arg3 : i32
  %2 = xls.send %tkn_out_0, %result_1, @proxy_arg1 : i32
  xls.yield %arg0 : i32
}
xls.chan @proxy_arg0 : i32
xls.chan @proxy_arg1 : i32
xls.chan @proxy_arg2 : i32
xls.chan @proxy_arg3 : i32
xls.instantiate_eproc @proxy_0 (@proxy_arg0 as @IntegrationTestLabel, @proxy_arg1 as @resp, @proxy_arg2 as @rom1_req, @proxy_arg3 as @rom1_resp)
xls.eproc @fetch_0(%arg0: i32) zeroinitializer discardable {
  %0 = xls.after_all  : !xls.token
  %1 = xls.send %0, %arg0, @fetch_arg0 : i32
  %tkn_out, %result = xls.blocking_receive %1, @fetch_arg1 : i32
  xls.yield %result : i32
}
xls.chan @fetch_arg0 : i32
xls.chan @fetch_arg1 : i32
xls.instantiate_eproc @fetch_0 (@fetch_arg0 as @IntegrationTestLabel, @fetch_arg1 as @resp)
xls.chan @boundary1 {send_supported = false} : i32
xls.chan @boundary2 {recv_supported = false} : i32
xls.instantiate_eproc @rom_0 (@rom_arg0 as @boundary1, @rom_arg1 as @boundary2)
