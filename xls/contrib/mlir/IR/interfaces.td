#ifndef HWIR_INTERFACES
#define HWIR_INTERFACES

include "mlir/IR/OpBase.td"

// OpInterface declaring that an op's region should be converted to XLS.
def XlsRegionOpInterface : OpInterface<"XlsRegionOpInterface"> {
  let description = [{
    Declares an op's region as being convertable to XLS ops.
  }];
  let cppNamespace = "::mlir::xls";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns true if the region should be converted",
      /*retTy=*/"bool",
      /*methodName=*/"isSupportedRegion",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return true;"
    >,
    InterfaceMethod<
      /*desc=*/"Returns the region to convert",
      /*retTy=*/"::mlir::Region&",
      /*methodName=*/"getBodyRegion",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/"Returns the name corresponding to the op. The name represents"
      "a symbol in the same scope as the op (e.g., like name of FuncOp). The"
      "name is up to the user/higher-level tool.",
      /*retTy=*/"::llvm::StringRef",
      /*methodName=*/"getName",
      /*args=*/(ins)
    >,
    StaticInterfaceMethod<
      /*desc=*/"Builds terminator for this region op.",
      /*retTy=*/"::mlir::Operation*",
      /*methodName=*/"buildTerminator",
      /*args=*/(ins "::mlir::Location":$loc,
                    "::mlir::OpBuilder&":$builder,
                    "::mlir::ValueRange":$operands
               )
    >,
    StaticInterfaceMethod<
      /*desc=*/"Adds signature conversion patterns and legality patterns",
      /*retTy=*/"void",
      /*methodName=*/"addSignatureConversionPatterns",
      /*args=*/(ins
                  "::mlir::RewritePatternSet&":$patterns,
                  "::mlir::TypeConverter&":$typeConverter,
                  "::mlir::ConversionTarget&":$target
               )
    >,
  ];
}

// Declares that an op is predicatable.
def PredicatableOpInterface : OpInterface<"PredicatableOpInterface"> {
  let description = [{
    Declares that an op is predicatable.
  }];
  let cppNamespace = "::mlir::xls";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the condition of the op. May return null.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getCondition",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/"Sets the condition of the op.",
      /*retTy=*/"void",
      /*methodName=*/"setCondition",
      /*args=*/(ins "Value":$condition)
    >,
  ];
}

// Declares that an op is permissible inside a literal region.
def LiteralMemberOpInterface : OpInterface<"LiteralMemberOpInterface"> {
  let description = [{
    Declares that an op is permissible inside a literal region.
  }];
  let cppNamespace = "::mlir::xls";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns true if the operation, in its current form, is permissible inside a literal region.",
      /*retTy=*/"bool",
      /*methodName=*/"isPermissibleInsideLiteral",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return true;"
    >,
  ];
}

#endif
