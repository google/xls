include "xls/contrib/mlir/IR/xls_ops.td"
include "mlir/Dialect/Math/IR/MathOps.td"
include "mlir/IR/PatternBase.td"

class StringAttr<string str> : ConstantStrAttr<StrAttr, str>;
// TODO(jpienaar): Using FirstResult below as work around as variadic result
// type is not being set and this is the root node.
def FirstResult : NativeCodeCall<"$0->getResult(0)">;

def : Pat<(Math_Exp2Op:$op $a, /*FastMathFlags=*/$_),
          (FirstResult (Xls_CallDslxOp
            StringAttr<"xls/compute_exp2.x">,
            StringAttr<"compute_exp2">, (variadic $a),
            ConstUnitAttr, ConstUnitAttr,
            (returnType $op)))>;
