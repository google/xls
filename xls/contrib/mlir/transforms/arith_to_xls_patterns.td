include "xls/contrib/mlir/IR/xls_ops.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir/Dialect/Math/IR/MathOps.td"
include "mlir/IR/PatternBase.td"

class BinaryOpPat<Arith_Op a, Xls_Op b> : Pat<(a $a, $b), (b $a, $b)>;
class BinaryOpOverflowPat<Arith_Op a, Xls_Op b> : Pat<(a $a, $b, $_), (b $a, $b)>;
class BinaryVariadicOpPat<Arith_Op a, Xls_Op b> : Pat<(a $a, $b), (b (variadic $a, $b))>;
class UnaryOpPat<Arith_Op a, Xls_Op b> : Pat<(a $a), (b $a)>;

def : BinaryOpOverflowPat<Arith_AddIOp, Xls_AddOp>;
// TODO(jmolloy): Are smul and umul the equivalent as we use "normal" two's
// complement representation?
def : BinaryOpOverflowPat<Arith_MulIOp, Xls_UmulOp>;
def : BinaryOpPat<Arith_DivSIOp, Xls_SdivOp>;
def : BinaryOpPat<Arith_RemSIOp, Xls_SmodOp>;
def : BinaryOpOverflowPat<Arith_SubIOp, Xls_SubOp>;
def : BinaryOpPat<Arith_DivUIOp, Xls_UdivOp>;
def : BinaryOpPat<Arith_RemUIOp, Xls_UmodOp>;
def : BinaryVariadicOpPat<Arith_AndIOp, Xls_AndOp>;
def : BinaryVariadicOpPat<Arith_OrIOp, Xls_OrOp>;
def : BinaryVariadicOpPat<Arith_XOrIOp, Xls_XorOp>;

def : UnaryOpPat<Arith_ExtSIOp, Xls_SignExtOp>;
def : UnaryOpPat<Arith_ExtUIOp, Xls_ZeroExtOp>;

class CmpPat<string s, Xls_Op target> :
    Pat<(Arith_CmpIOp ConstantEnumCase<Arith_CmpIPredicateAttr, s>, $lhs, $rhs),
        (target $lhs, $rhs)>;

def : CmpPat<"eq", Xls_EqOp>;
def : CmpPat<"ne", Xls_NeOp>;
def : CmpPat<"sge", Xls_SgeOp>;
def : CmpPat<"sgt", Xls_SgtOp>;
def : CmpPat<"sle", Xls_SleOp>;
def : CmpPat<"slt", Xls_SltOp>;
def : CmpPat<"uge", Xls_UgeOp>;
def : CmpPat<"ugt", Xls_UgtOp>;
def : CmpPat<"ule", Xls_UleOp>;
def : CmpPat<"ult", Xls_UltOp>;

def : BinaryOpOverflowPat<Arith_ShLIOp, Xls_ShllOp>;
def : BinaryOpPat<Arith_ShRUIOp, Xls_ShrlOp>;
def : BinaryOpPat<Arith_ShRSIOp, Xls_ShraOp>;

class IntAttrBase<Pred predicate, string descr>
  : TypedAttrBase<AnyInteger, "IntegerAttr",
    And<[CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">,
         predicate]>,
    "integer attribute" # descr> {
  let returnType = [{ ::llvm::APInt }];
}

def AnyFloatAttr
  : TypedAttrBase<AnyFloat, "FloatAttr",
    CPred<"::llvm::isa<::mlir::FloatAttr>($_self)">,
    "float attribute"> {
  let returnType = [{ ::llvm::APFloat }];
}

def AnyFloatElementsAttr : ElementsAttrBase<
  CPred<"::llvm::isa<::mlir::DenseFPElementsAttr>($_self)">,
        "float elements attribute"> {
  let storageType = [{ ::mlir::DenseElementsAttr }];
  let returnType = [{ ::mlir::DenseElementsAttr }];

  let convertFromStorage = "$_self";
}

def IntNonIndexAttr
  : IntAttrBase<Neg<IndexAttr.predicate>,
                " whose value is not of type index">;

def NonIndexAttr : AnyAttrOf<[IntNonIndexAttr, AnyFloatAttr]>;

def IntNonIndexElementsAttr
    : IntElementsAttrBase<Neg<IndexElementsAttr.predicate>,
                          " whose elements are not of type index">;

def NonIndexElementsAttr : AnyAttrOf<[IntNonIndexElementsAttr,
                                      AnyFloatElementsAttr]>;

def : Pat<(Arith_ConstantOp NonIndexElementsAttr:$val),
          (Xls_ConstantTensorOp $val)>;
def : Pat<(Arith_ConstantOp NonIndexAttr:$val),
          (Xls_ConstantScalarOp $val)>;

def FloatLibcallSymbol :
    NativeCodeCall<"getOrCreateFloatLibcallSymbol($_builder, $0, $1)">;
def FloatLibcallSymbolReturningBool :
    NativeCodeCall<"getOrCreateFloatLibcallSymbol($_builder, $0, $1, true)">;
def FloatLibcallSymbol_ExtTrunc :
    NativeCodeCall<"getOrCreateFloatLibcallSymbol($_builder, $0, $1, false, true)">;

def FirstResult : NativeCodeCall<"$0->getResult(0)">;

class FloatLibcall<Op Op, string Name> :
    Pat<(Op:$op $a, $b, /*FastMathFlags=*/$_),
          (FirstResult
            (VectorizedCallOp
              (FloatLibcallSymbol ConstantStrAttr<StrAttr, Name>, $op),
              (variadic $a, $b)))>;

def : FloatLibcall<Arith_AddFOp, "add">;
def : FloatLibcall<Arith_MulFOp, "mul">;
def : FloatLibcall<Arith_SubFOp, "sub">;

def : Pat<(Arith_ExtFOp:$op $a, /*FastMathFlags=*/$_),
          (FirstResult
            (VectorizedCallOp
              (FloatLibcallSymbol_ExtTrunc ConstantStrAttr<StrAttr, "ext">, $op),
              (variadic $a)))>;

def : Pat<(Arith_TruncFOp:$op $a, /*RoundingMode=*/$_, /*FastMathFlags=*/$_),
          (FirstResult
            (VectorizedCallOp
              (FloatLibcallSymbol_ExtTrunc ConstantStrAttr<StrAttr, "trunc">, $op),
              (variadic $a)))>;

// The expansion is a little tricky to read due to the one-hot select with the
// default case being the first argument.
// (> a b)           -> 0b1 if a > b else 0b0
// (sel (> a b) ...) -> first case if (a > b) else default case.
// (sel (> a b) b a) -> a if (a > b) else b (b is default case).
class MinMaxPatBase<dag Matcher, dag Predicate> : Pat<
  Matcher,
  (Xls_SelOp Predicate, /*otherwise=*/$b, /*cases=*/(variadic $a))>;

class FPMinMaxPat<Op Op, string Name> : MinMaxPatBase<
  (Op:$op $a, $b, /*FastMathFlags=*/$_),
  (FirstResult
    (VectorizedCallOp
      (FloatLibcallSymbolReturningBool ConstantStrAttr<StrAttr, Name>, $op),
      (variadic $a, $b)))>;

def : FPMinMaxPat<Arith_MaximumFOp, "gt_2">;
def : FPMinMaxPat<Arith_MinimumFOp, "lt_2">;

def : MinMaxPatBase<(Arith_MaxSIOp:$op $a, $b), (Xls_SgtOp $a, $b)>;
def : MinMaxPatBase<(Arith_MinSIOp:$op $a, $b), (Xls_SltOp $a, $b)>;
def : MinMaxPatBase<(Arith_MaxUIOp:$op $a, $b), (Xls_UgtOp $a, $b)>;
def : MinMaxPatBase<(Arith_MinUIOp:$op $a, $b), (Xls_UltOp $a, $b)>;

def : Pat<(SelectOp $selector, $a, $b),
          (Xls_SelOp $selector, /*otherwise=*/$b, /*cases=*/(variadic $a))>;

def GetBitwidthAsI64Attr : NativeCodeCall<
  "$_builder.getI64IntegerAttr(cast<IntegerType>($0.getType()).getWidth())"
>;

def : Pat<
  (Arith_TruncIOp:$op $a),
  (Xls_BitSliceOp
    $a,
    (NativeCodeCall<"$_builder.getI64IntegerAttr(0)">),
    (GetBitwidthAsI64Attr $op)
  )
>;

// TODO(jmolloy): Support Div and Exp in the standard library.
def : FloatLibcall<Arith_DivFOp, "add">;
def : Pat<(Math_ExpOp:$op $a, /*FastMathFlags=*/$_),
          (FirstResult
            (VectorizedCallOp
              (FloatLibcallSymbol ConstantStrAttr<StrAttr, "ceil">, $op),
              (variadic $a)))>;
