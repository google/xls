#ifndef MLIR_XLS_TRANSFORMS_PASSES
#define MLIR_XLS_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ArithToXlsPass : Pass<"arith-to-xls", "::mlir::ModuleOp"> {
  let summary = "Converts Arith ops to XLS ops";
  let dependentDialects = [
    "mlir::xls::XlsDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::math::MathDialect"
  ];
}

def LowerCountedForPass : Pass<"xls-lower-for-ops", "::mlir::ModuleOp"> {
  let summary = "Lowers ForOp to CountedForOp.";
  let dependentDialects = [
    "mlir::xls::XlsDialect",
  ];
}

def MathToXlsPass : Pass<"math-to-xls", "::mlir::ModuleOp"> {
  let summary = "Lower Math operations to XLS operations";
  let description = [{
    Pass that converts Math operations to the equivalent operations using
    XLS operations and DSLX calls.
  }];
  let dependentDialects = ["mlir::xls::XlsDialect"];
}

def NormalizeXlsCallsPass : Pass<"normalize-xls-calls", "::mlir::ModuleOp"> {
  let summary = "Normalize XLS calls across packages";
  let description = [{
    Normalizes XLS calls to reduce number of package import. Post this
    pass there should be no XLS Call ops and only package imports,
    functions with linkage and calls.
  }];
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def ScalarizePass : Pass<"scalarize", "::mlir::ModuleOp"> {
  let summary = "Scalarizes tensor XLS operations";
  let dependentDialects = [
    "mlir::xls::XlsDialect",
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def ScfToXlsPass : Pass<"scf-to-xls", "::mlir::ModuleOp"> {
  let summary = "Converts SCF ops to XLS ops";
  let dependentDialects = [
    "mlir::xls::XlsDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect"
  ];
}

def IndexTypeConversionPass : Pass<"index-type-conversion", "mlir::ModuleOp"> {
  let summary = "Convert index type to int type for any ops";
  let description = [{
    Pass that converts index type to int type for any ops.
  }];
  let dependentDialects = [
    "mlir::xls::XlsDialect",
    "mlir::arith::ArithDialect",
  ];
  let options = [
    Option<"indexTypeBitWidth", "index-bitwidth", "unsigned",
    /*default=*/"32",
    "Specify the index type bit width">
  ];
}

def ProcElaborationPass : Pass<"elaborate-procs", "mlir::ModuleOp"> {
  let summary = "Elaborates sprocs into eprocs";
  let description = [{
    Pass that lowers (elaborates) sprocs into eprocs and instantiate_eproc ops.
  }];
  let dependentDialects = [
    "mlir::xls::XlsDialect",
  ];
}

def InstantiateEprocsPass : Pass<"instantiate-eprocs", "mlir::ModuleOp"> {
  let summary = "Instantiates eprocs";
  let description = [{
    Pass that lowers instantiate_eproc ops into eprocs.
  }];
  let dependentDialects = [
    "mlir::xls::XlsDialect",
  ];
}

def ExpandMacroOpsPass : Pass<"expand-macro-ops", "mlir::ModuleOp"> {
  let summary = "Expands macro ops such as ArrayUpdateSlice";
  let description = [{
    Pass that expands macro ops that are unsupported by translation.
  }];
  let dependentDialects = [
    "mlir::xls::XlsDialect",
  ];
}

def ArrayToBitsPass : Pass<"array-to-bits", "mlir::ModuleOp"> {
  let summary = "Converts arrays to bitstrings";
  let description = [{
    Pass that converts array types to bitstrings (IntegerTypes). Must be run
    after scalarization.

    Use of this transform can help larger modules get through XLS optimizations
    and verilog emission easier, but it should be used sparingly as XLS should
    perform better optimizations on arrays.
  }];
  let dependentDialects = [
    "mlir::xls::XlsDialect",
  ];
}

#endif // MLIR_XLS_TRANSFORMS_PASSES
