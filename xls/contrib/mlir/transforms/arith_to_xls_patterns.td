include "xls/contrib/mlir/IR/xls_ops.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir/Dialect/Math/IR/MathOps.td"
include "mlir/IR/PatternBase.td"

def FloatLib : NativeCodeCall<"getFloatLib($0.getType())">;

// Shorthand for a constant string attribute.
class CS<string s> : ConstantStrAttr<StrAttr, s>;

class ScalarOrTensorOf<Type element> :
    AnyTypeOf<[element, TensorOf<[element]>]>;

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

class FCmpPat<string s, dag target> :
    Pat<(Arith_CmpFOp:$op ConstantEnumCase<Arith_CmpFPredicateAttr, s>, $lhs, $rhs, $fmf),
        target>;

def : FCmpPat<"oeq", (Xls_CallDslxOp (FloatLib $lhs), CS<"eq_2">,
                                     (variadic $lhs, $rhs), ConstUnitAttr)>;
def : FCmpPat<"ogt", (Xls_CallDslxOp (FloatLib $lhs), CS<"gt_2">,
                                     (variadic $lhs, $rhs), ConstUnitAttr)>;
def : FCmpPat<"oge", (Xls_CallDslxOp (FloatLib $lhs), CS<"gte_2">,
                                     (variadic $lhs, $rhs), ConstUnitAttr)>;
def : FCmpPat<"olt", (Xls_CallDslxOp (FloatLib $lhs), CS<"lt_2">,
                                     (variadic $lhs, $rhs), ConstUnitAttr)>;
def : FCmpPat<"ole", (Xls_CallDslxOp (FloatLib $lhs), CS<"lte_2">,
                                     (variadic $lhs, $rhs), ConstUnitAttr)>;

// Define unordered(x, y) = is_nan(x + y)
def : FCmpPat<"uno", (Xls_CallDslxOp (FloatLib $lhs), CS<"is_nan">,
                                     (variadic (Arith_AddFOp $lhs, $rhs, $fmf)),
                                     ConstUnitAttr)>;

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

// The extension and truncation library.
defvar ExtLib = CS<"xls/contrib/mlir/stdlib/fp_ext_trunc.x">;

def : Pat<(Arith_NegFOp:$op $a, /*FastMathFlags=*/$_),
  (Xls_CallDslxOp (FloatLib $a), CS<"negate">, (variadic $a), ConstUnitAttr)>;

// Emits a binary float library call.
class FloatLibcall<Op Op, string Name> :
    Pat<(Op:$op $a, $b, /*FastMathFlags=*/$_),
        (Xls_CallDslxOp (FloatLib $a), CS<Name>, (variadic $a, $b), ConstUnitAttr)>;

def : FloatLibcall<Arith_AddFOp, "add">;
def : FloatLibcall<Arith_MulFOp, "mul">;
def : FloatLibcall<Arith_SubFOp, "sub">;

def : Pat<(Arith_ExtFOp:$op $a, /*FastMathFlags=*/$_),
          (Xls_CallDslxOp ExtLib, CS<"ext">, (variadic $a), ConstUnitAttr)>;

def : Pat<(Arith_TruncFOp:$op $a, /*RoundingMode=*/$_, /*FastMathFlags=*/$_),
          (Xls_CallDslxOp ExtLib, CS<"trunc">, (variadic $a), ConstUnitAttr)>;

def : Pat<(Arith_SIToFPOp:$op I32:$a),
          (Xls_CallDslxOp (FloatLib $op), CS<"from_int32">, (variadic $a), ConstUnitAttr),
          [(ScalarOrTensorOf<F32> $op)]>;

def : Pat<(Arith_SIToFPOp:$op I32:$a),
          (Xls_CallDslxOp (FloatLib $op), CS<"from_float32">,
            (variadic (Arith_SIToFPOp $a, (returnType "$_builder.getF32Type()"))), ConstUnitAttr),
          [(ScalarOrTensorOf<BF16> $op)]>;

def : Pat<(Arith_SIToFPOp:$op I8:$a),
          (Xls_CallDslxOp (FloatLib $op), CS<"from_int8">, (variadic $a), ConstUnitAttr),
          [(ScalarOrTensorOf<BF16> $op)]>;

def : Pat<(Arith_FPToSIOp:$op F32:$a),
          (Xls_CallDslxOp (FloatLib $a), CS<"to_int32">, (variadic $a), ConstUnitAttr),
          [(ScalarOrTensorOf<I32> $op)]>;

def : Pat<(Arith_FPToSIOp:$op BF16:$a),
          (Xls_CallDslxOp (FloatLib $a), CS<"to_int16">, (variadic $a), ConstUnitAttr),
          [(ScalarOrTensorOf<I16> $op)]>;

// TODO(jmolloy): to_int8 doesn't exist, so truncating the result of to_int16
// seems like a reasonable approximation but I don't know if it's bit accurate.
def : Pat<(Arith_FPToSIOp:$op BF16:$a),
          (Arith_TruncIOp
            (Xls_CallDslxOp (FloatLib $a), CS<"to_int16">, (variadic $a),
                            ConstUnitAttr, (returnType "$_builder.getI16Type()"))),
          [(ScalarOrTensorOf<I8> $op)]>;

// Creates a `SelOp` with two `cases` (and no `otherwise` operand). This wrapper
// is needed because DRR doesn't deal well with the `AttrSizedOperandSegments`
// of the `SelOp`.
// NB: the order of the cases follows the xls::SelOp convention, i.e. 0 (false)
// yields the first case, 1 (true) yields the second case.
def createSelOp2Cases : NativeCodeCall<[{
    $_builder.create<::mlir::xls::SelOp>(
        $_loc, $0, ArrayRef<::mlir::Value>{$1, $2})
  }]>;

// (> a b)           -> 0b1 if a > b else 0b0
// (sel (> a b) ...) -> first case if (a > b) else second case.
// (sel (> a b) b a) -> a if (a > b) else b.
class MinMaxPatBase<dag Matcher, dag Predicate> : Pat<
  Matcher,
  (createSelOp2Cases Predicate, $b, $a)>;

class FPMinMaxPat<Op Op, string Name> : MinMaxPatBase<
  (Op:$op $a, $b, /*FastMathFlags=*/$_),
  (Xls_CallDslxOp (FloatLib $a), CS<Name>, (variadic $a, $b), ConstUnitAttr,
     (returnType "boolLike(op)"))>;

def : FPMinMaxPat<Arith_MaximumFOp, "gt_2">;
def : FPMinMaxPat<Arith_MinimumFOp, "lt_2">;

def : MinMaxPatBase<(Arith_MaxSIOp:$op $a, $b), (Xls_SgtOp $a, $b)>;
def : MinMaxPatBase<(Arith_MinSIOp:$op $a, $b), (Xls_SltOp $a, $b)>;
def : MinMaxPatBase<(Arith_MaxUIOp:$op $a, $b), (Xls_UgtOp $a, $b)>;
def : MinMaxPatBase<(Arith_MinUIOp:$op $a, $b), (Xls_UltOp $a, $b)>;

def : Pat<(SelectOp $selector, $a, $b), (createSelOp2Cases $selector, $b, $a)>;

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
