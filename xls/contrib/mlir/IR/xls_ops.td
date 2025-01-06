#ifndef HWIR_XLS_OPS
#define HWIR_XLS_OPS

include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "xls/contrib/mlir/IR/interfaces.td"

def Xls_Dialect : Dialect {
  let name = "xls";
  let cppNamespace = "mlir::xls";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let hasConstantMaterializer = 1;
}

class Xls_Op<string mnemonic, list<Trait> traits = []> :
    Op<Xls_Dialect, mnemonic, traits>;

class Xls_Type<string name, list<Trait> traits = [],
               string baseCppClass = "::mlir::Type">
    : TypeDef<Xls_Dialect, name, traits, baseCppClass> {}

def Xls_TokenType : Xls_Type<"Token"> {
  let summary = "Token used for enforcing ordering between channel operations";
  let description = [{
    A type used to enforce ordering between channel operations. The token type
    has no value and all tokens are identical. A token is purely symbolic /
    semantic and has no correlate in hardware.
  }];
  let mnemonic = "token";
}

def Xls_ArrayType : Xls_Type<"Array", [ShapedTypeInterface]> {
  let summary = "array";
  let mnemonic = "array";
  let parameters = (ins
    TypeParameter<"int", "Number of elements">:$num_elements,
    TypeParameter<"::mlir::Type", "Underlying type">:$element_type
  );
  let assemblyFormat = "`<` $num_elements `x` $element_type `>`";
  let builders = [
    TypeBuilderWithInferredContext<(ins "int":$num_elements,
                                        "::mlir::Type":$element_type), [{
      return Base::get(element_type.getContext(), num_elements, element_type);
    }]>
  ];
  let extraClassDeclaration = [{
    constexpr bool hasRank() { return true; }
    ::llvm::SmallVector<int64_t> getShape() const {
      return ::llvm::SmallVector<int64_t>(1, getNumElements());
    }
    ::mlir::ShapedType cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape,
                                 Type elementType) {
      assert(0 && "ArrayType::cloneWith unimplemented!");
      return {};
    }
  }];
}

def Xls_OpaqueType : Xls_Type<"Opaque"> {
  let summary = "Opaque type used for interfacing with DSLX";
  let description = [{
    A type used to interface with DSLX. The opaque type allows DSLX function
    parameter and return types to be type-erased.

    The only way of producing an opaque type is by calling a DSLX function.

    Opaque types are parametric on an arbitrary string to aid typechecking.
  }];
  let mnemonic = "opaque";
  let parameters = (ins
    TypeParameter<"::mlir::StringAttr", "Tag">:$tag
  );
  let assemblyFormat = "`<` $tag `>`";
}

// An operation that can be applied to either tensors or arrays. Used for
// packing/unpacking operations like ArrayOp and TupleIndexOp that do not care
// about the nested type of their operands.
def TensorArrayTypeFungible : NativeOpTrait<"TensorArrayTypeFungible"> {
  let cppNamespace = "::mlir::xls";
}

// We represent bits as signless integers. We allow tensors of bits, these will
// be flattened during code generation.
defvar Xls_ScalarBits = AnyTypeOf<[AnySignlessInteger, Index, AnyFloat]>;
defvar Xls_Bits = AnyTypeOf<[StaticShapeTensorOf<[Xls_ScalarBits]>,
                             Xls_ArrayType,
                             Xls_ScalarBits,
                             Xls_OpaqueType]>;
defvar Xls_BitsOrTuple = AnyTypeOf<[Xls_Bits, NestedTupleOf<[Xls_Bits]>]>;

// Base class for XLS dialect attributes.
class Xls_Attr<string name, list<Trait> traits = []>
    : AttrDef<Xls_Dialect, name, traits> {
  let cppClassName = name;
  let mnemonic = ?;
}

// Perform one of two kinds of linkage:
// - import - import to XLS and allow co-optimization;
// - foreign - import to XLS but tag as foreign/do not allow co-optimization;
def Xls_LinkageKindImport : I32EnumAttrCase<"kImport", 0, "import">;
def Xls_LinkageKindForeign : I32EnumAttrCase<"kForeign", 1, "foreign">;
def Xls_LinkageKind : I32EnumAttr<"LinkageKind", "XLS linkage kind",
  [Xls_LinkageKindImport, Xls_LinkageKindForeign]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::xls";
}
def Xls_LinkageKindAttr : EnumAttr<Xls_Dialect, Xls_LinkageKind, "linkage_kind">;

def Xls_TranslationLinkage : Xls_Attr<"TranslationLinkage"> {
  let summary = "Translation linkage attribute";
  let description = [{
    Refers to a DSLX package and symbol to resolve too at translation time.
  }];
  let parameters = (ins
    "::mlir::SymbolRefAttr":$package,
    "::mlir::StringAttr":$function,
    DefaultValuedParameter<"mlir::xls::LinkageKind", "LinkageKind::kImport">:$kind);
  let mnemonic = "translation_linkage";
  let assemblyFormat = "`<` $package `:` $function (`,` $kind^)? `>`";
  let attrName = "xls.translation_linkage";
}

class GetShapeSplat<string name> :
    StrFunc<"getShapeSplat($" # name # ".getType())">;

// Validates that the shape of all of `names` are the same. Handles scalar and
// shaped types. Handles variadics.
class ShapesAreConsistent<list<string> names> :
    AllMatchSameOperatorTrait<names, GetShapeSplat<"_self">.result, "shape">;

class Xls_UnaryOp<string name, list<Trait> traits = []> :
  Xls_Op<name, !listconcat(traits, [
    ShapesAreConsistent<["operand", "result"]>,
    Elementwise,
    Scalarizable,
  ])> {
  let assemblyFormat = [{
    $operand attr-dict `:` custom<SameOperandsAndResultType>(type($operand), type($result))
  }];
}

class Xls_BinaryOp<string name, list<Trait> traits = []> :
  Xls_Op<name, !listconcat(traits, [
    ShapesAreConsistent<["lhs", "rhs", "result"]>,
    Elementwise,
    Scalarizable,
  ])> {
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` custom<SameOperandsAndResultType>(type($lhs), type($rhs), type($result))
  }];
}

class Xls_NaryOp<string name, list<Trait> traits = []> :
  Xls_Op<name, !listconcat(traits, [
    ShapesAreConsistent<["inputs", "result"]>,
    Elementwise,
    Scalarizable
  ])> {
  let assemblyFormat = [{
    $inputs attr-dict `:` custom<VariadicSameOperandsAndResultType>(ref($inputs), type($inputs), type($result))
  }];
  let builders = [
    OpBuilder<(ins "::mlir::ValueRange":$inputs), [{
      build($_builder, $_state, inputs.front().getType(), inputs);
    }]>,
    // Common case of a binary op.
    OpBuilder<(ins "::mlir::Value":$lhs, "::mlir::Value":$rhs), [{
      ::mlir::ValueRange inputs = {lhs, rhs};
      build($_builder, $_state, lhs.getType(), inputs);
    }]>
  ];
}

class Xls_NonScalarizableNaryOp<string name, list<Trait> traits = []> :
  Xls_Op<name, !listconcat(traits, [
    ShapesAreConsistent<["inputs", "result"]>,
  ])> {
  let assemblyFormat = [{
    $inputs attr-dict `:` custom<VariadicSameOperandsAndResultType>(ref($inputs), type($inputs), type($result))
  }];
}

//===----------------------------------------------------------------------===//
// Unary bitwise operations
//===----------------------------------------------------------------------===//

def Xls_IdentityOp : Xls_UnaryOp<"identity", [Pure, SameOperandsAndResultType]> {
  let summary = "Identity operation";
  let description = [{
    result = operand
  }];
  let arguments = (ins
    Xls_Bits:$operand
  );
  let results = (outs
    Xls_Bits:$result
  );
}

def Xls_NotOp : Xls_UnaryOp<"not", [Pure, SameOperandsAndResultType]> {
  let summary = "Complement operation";
  let description = [{
    result = ~operand
  }];
  let arguments = (ins
    Xls_Bits:$operand
  );
  let results = (outs
    Xls_Bits:$result
  );
}

//===----------------------------------------------------------------------===//
// Variadic bitwise operations
//===----------------------------------------------------------------------===//

def Xls_AndOp : Xls_NaryOp<"and", [Commutative, Pure, SameOperandsAndResultType]> {
  let summary = "Bitwise AND operation";
  let description = [{
    result = operand_0 & ... & operand_{N-1}
  }];
  let arguments = (ins
    Variadic<Xls_Bits>:$inputs
  );
  let results = (outs
    Xls_Bits:$result
  );
}

def Xls_OrOp : Xls_NaryOp<"or", [Commutative, Pure, SameOperandsAndResultType]> {
  let summary = "Bitwise OR operation";
  let description = [{
    result = operand_0 | ... | operand_{N-1}
  }];
  let arguments = (ins
    Variadic<Xls_Bits>:$inputs
  );
  let results = (outs
    Xls_Bits:$result
  );
}

def Xls_XorOp : Xls_NaryOp<"xor", [Commutative, Pure, SameOperandsAndResultType]> {
  let summary = "Bitwise XOR operation";
  let description = [{
    result = operand_0 ^ ... ^ operand_{N-1}
  }];
  let arguments = (ins
    Variadic<Xls_Bits>:$inputs
  );
  let results = (outs
    Xls_Bits:$result
  );
}

//===----------------------------------------------------------------------===//
// Arithmetic unary operations
//===----------------------------------------------------------------------===//

def Xls_NegOp : Xls_UnaryOp<"neg", [Pure, SameOperandsAndResultType]> {
  let summary = "Integer negation operation";
  let description = [{
    result = -operand
  }];
  let arguments = (ins
    Xls_Bits:$operand
  );
  let results = (outs
    Xls_Bits:$result
  );
}

//===----------------------------------------------------------------------===//
// Arithmetic binary operations
//===----------------------------------------------------------------------===//

class Xls_ArithBinaryOp<string name, string desc>
    : Xls_BinaryOp<name, [Pure, SameOperandsAndResultType]> {
  let summary = !strconcat(desc, " operation");
  let description = summary;
  let arguments = (ins
    Xls_Bits:$lhs,
    Xls_Bits:$rhs
  );
  let builders = [
    OpBuilder<(ins "::mlir::Value":$lhs, "int":$rhs), [{
      ::mlir::Value rhs_val = $_builder.create<ConstantScalarOp>(lhs.getLoc(), lhs.getType(), $_builder.getI32IntegerAttr(rhs));
      build($_builder, $_state, lhs, rhs_val);
    }]>
  ];
  let results = (outs
    Xls_Bits:$result
  );
}

def Xls_AddOp : Xls_ArithBinaryOp<"add", "addition"> {
  let hasFolder = 1;
}
def Xls_SmulOp : Xls_ArithBinaryOp<"smul", "signed multiplication">;
def Xls_UmulOp : Xls_ArithBinaryOp<"umul", "unsigned multiplication"> {
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}
def Xls_SdivOp : Xls_ArithBinaryOp<"sdiv", "signed division">;
def Xls_SmodOp : Xls_ArithBinaryOp<"smod", "signed modulo">;
def Xls_SubOp : Xls_ArithBinaryOp<"sub", "subtraction">;
def Xls_UdivOp : Xls_ArithBinaryOp<"udiv", "unsigned division">;
def Xls_UmodOp : Xls_ArithBinaryOp<"umod", "unsigned modulo">;

// TODO(jmolloy): Smulp/Umulp - partial product multiplication.

//===----------------------------------------------------------------------===//
// Comparison operations
//===----------------------------------------------------------------------===//

class Xls_ComparisonOp<string name, Type operand_type>
    : Xls_BinaryOp<name, [Pure, SameTypeOperands]> {
  let summary = !strconcat(name, " comparison");
  let description = summary;
  let arguments = (ins
    operand_type:$lhs,
    operand_type:$rhs
  );
  let results = (outs
    Xls_Bits:$result
  );
  let builders = [
    OpBuilder<(ins "::mlir::Value":$lhs, "::mlir::Value":$rhs), [{
      build($_builder, $_state, getI1TypeOf(lhs.getType()), lhs, rhs);
    }]>,
    OpBuilder<(ins "::mlir::Value":$lhs, "int":$rhs), [{
      ::mlir::Value rhs_val = $_builder.create<ConstantScalarOp>(lhs.getLoc(), lhs.getType(), $_builder.getI32IntegerAttr(rhs));
      build($_builder, $_state, getI1TypeOf(lhs.getType()), lhs, rhs_val);
    }]>,
  ];
}

def Xls_EqOp : Xls_ComparisonOp<"eq", Xls_BitsOrTuple>;
def Xls_NeOp : Xls_ComparisonOp<"ne", Xls_BitsOrTuple>;
def Xls_SgeOp : Xls_ComparisonOp<"sge", Xls_Bits>;
def Xls_SgtOp : Xls_ComparisonOp<"sgt", Xls_Bits>;
def Xls_SleOp : Xls_ComparisonOp<"sle", Xls_Bits>;
def Xls_SltOp : Xls_ComparisonOp<"slt", Xls_Bits>;
def Xls_UgeOp : Xls_ComparisonOp<"uge", Xls_Bits>;
def Xls_UgtOp : Xls_ComparisonOp<"ugt", Xls_Bits>;
def Xls_UleOp : Xls_ComparisonOp<"ule", Xls_Bits>;
def Xls_UltOp : Xls_ComparisonOp<"ult", Xls_Bits>;

//===----------------------------------------------------------------------===//
// Shift operations
//===----------------------------------------------------------------------===//

class Xls_ShiftOp<string name, string desc>
    : Xls_BinaryOp<name, [Pure, AllTypesMatch<["lhs", "result"]>]> {
  let summary = !strconcat(desc, " operation");
  let description = summary;
  let arguments = (ins
    Xls_Bits:$lhs,
    Xls_Bits:$rhs
  );
  let results = (outs
    Xls_Bits:$result
  );
}

def Xls_ShllOp : Xls_ShiftOp<"shll", "Shift left logical">;
def Xls_ShraOp : Xls_ShiftOp<"shra", "Shift right arithmetic">;
def Xls_ShrlOp : Xls_ShiftOp<"shrl", "Shift right logical">;

//===----------------------------------------------------------------------===//
// Extension operations
//===----------------------------------------------------------------------===//

def Xls_ZeroExtOp : Xls_UnaryOp<"zero_ext", [Pure, SameOperandsAndResultShape]> {
  let summary = "Zero extend";
  let description = [{
    Zero-extends a value: turns its bit-length into the new target bit-length by
    filling zeroes in the most significant bits.
  }];
  let arguments = (ins
    Xls_Bits:$operand
  );
  let results = (outs
    Xls_Bits:$result
  );
}

def Xls_SignExtOp : Xls_UnaryOp<"sign_ext", [Pure, SameOperandsAndResultShape]> {
  let summary = "Sign extend";
  let description = [{
    Sign-extends a value: turns its bit-length into the new target bit-length by
    filling in the most significant bits (MSbs) with the following policy:

      * ones in the MSbs if the MSb of the original value was set, or
      * zeros in the MSbs if the MSb of the original value was unset.
  }];
  let arguments = (ins
    Xls_Bits:$operand
  );
  let results = (outs
    Xls_Bits:$result
  );
}

//===----------------------------------------------------------------------===//
// Channel operations
//===----------------------------------------------------------------------===//

def Xls_BlockingReceiveOp : Xls_Op<"blocking_receive", [
  TensorArrayTypeFungible,
  PredicatableOpInterface,
  CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Receives a data value from a specified channel.";
  let description = [{
    Receives a data value from a specified channel. The type of the data value
    is determined by the channel. An optional predicate value conditionally
    enables the receive operation. A blocking receive waits (or blocks) until
    valid data is present at the channel.

    If the predicate is false the data values in the result are zero-filled.
  }];
  let arguments = (ins
    Xls_TokenType:$tkn,
    Optional<I1>:$predicate,
    SymbolRefAttr:$channel
  );
  let results = (outs
    Xls_TokenType:$tkn_out,
    Xls_BitsOrTuple:$result
  );
  let assemblyFormat = [{
    $tkn `,` ($predicate^ `,`)? $channel attr-dict `:` type($result)
  }];
  let extraClassDeclaration = [{
    ::mlir::Value getCondition() {
      return getPredicate();
    }
    void setCondition(::mlir::Value condition) {
      getPredicateMutable().assign(condition);
    }
    ::mlir::CallInterfaceCallable getCallableForCallee() {
      return getChannel();
    }
    void setCalleeFromCallable(::mlir::CallInterfaceCallable callable) {
      llvm_unreachable("setCalleeFromCallable is not supported");
    }
    ::mlir::Operation::operand_range getArgOperands() {
      return {getOperands().end(), getOperands().end()};
    }
    ::mlir::MutableOperandRange getArgOperandsMutable() { return {getOperation()}; }
  }];
}

def Xls_NonblockingReceiveOp : Xls_Op<"nonblocking_receive", [
  TensorArrayTypeFungible,
  PredicatableOpInterface,
  CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Receives a data value from a specified channel.";
  let description = [{
    Receives a data value from a specified channel. The type of the data value
    is determined by the channel. An optional predicate value conditionally
    enables the receive operation. Compared to a blocking receive, a
    non-blocking receive has an additional entry in its return tuple of type
    bits[1] denoting whether the data read is valid.

    If the predicate is false the data values in the result are zero-filled.
  }];
  let arguments = (ins
    Xls_TokenType:$tkn,
    Optional<I1>:$predicate,
    SymbolRefAttr:$channel
  );
  let results = (outs
    Xls_TokenType:$tkn_out,
    Xls_BitsOrTuple:$result,
    I1:$valid
  );
  let assemblyFormat = [{
    $tkn `,` ($predicate^ `,`)? $channel attr-dict `:` type($result)
  }];
  let extraClassDeclaration = [{
    ::mlir::Value getCondition() {
      return getPredicate();
    }
    void setCondition(::mlir::Value condition) {
      getPredicateMutable().assign(condition);
    }
    ::mlir::CallInterfaceCallable getCallableForCallee() {
      return getChannel();
    }
    void setCalleeFromCallable(::mlir::CallInterfaceCallable callable) {
      llvm_unreachable("setCalleeFromCallable is not supported");
    }
    ::mlir::Operation::operand_range getArgOperands() {
      return {getOperands().end(), getOperands().end()};
    }
    ::mlir::MutableOperandRange getArgOperandsMutable() {
      return {getOperation()};
    }
  }];
}

def Xls_SendOp : Xls_Op<"send", [
  TensorArrayTypeFungible,
  PredicatableOpInterface,
  CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Sends data to a specified channel.";
  let description = [{
    Sends data to a specified channel. The type of the data values is determined
    by the channel. An optional predicate value conditionally enables the send
    operation.

    The type of data must match the type supported by the channel.
  }];
  let arguments = (ins
    Xls_TokenType:$tkn,
    Xls_BitsOrTuple:$data,
    Optional<I1>:$predicate,
    SymbolRefAttr:$channel
  );
  let results = (outs
    Xls_TokenType:$result
  );
  let assemblyFormat = [{
    $tkn `,` $data `,` ($predicate^ `,`)? $channel attr-dict `:` type($data)
  }];
  let extraClassDeclaration = [{
    ::mlir::Value getCondition() {
      return getPredicate();
    }
    void setCondition(::mlir::Value condition) {
      getPredicateMutable().assign(condition);
    }
    ::mlir::CallInterfaceCallable getCallableForCallee() {
      return getChannel();
    }
    void setCalleeFromCallable(::mlir::CallInterfaceCallable callable) {
      llvm_unreachable("setCalleeFromCallable is not supported");
    }
    ::mlir::Operation::operand_range getArgOperands() {
      return {getOperands().end(), getOperands().end()};
    }
    ::mlir::MutableOperandRange getArgOperandsMutable() { return {getOperation()}; }
  }];
}

//===----------------------------------------------------------------------===//
// Array operations
//===----------------------------------------------------------------------===//

def Xls_ArrayOp : Xls_NonScalarizableNaryOp<"array", [Pure, SameTypeOperands, TensorArrayTypeFungible]> {
  let summary = "Array creation operation";
  let description = [{
    Constructs an array of its operands.
  }];
  let arguments = (ins
    Variadic<AnyType>:$inputs
  );
  let results = (outs
    Xls_ArrayType:$result
  );
}

def Xls_ArrayZeroOp : Xls_Op<"array_zero", [Pure, TensorArrayTypeFungible]> {
  let summary = "Array creation operation from a zero constant";
  let description = [{
    Constructs an array by splatting the zero constant.
  }];
  let results = (outs
    Xls_ArrayType:$result
  );
}

def Xls_ArrayIndexOp : Xls_Op<"array_index", [Pure, ShapesAreConsistent<["array", "result"]>,
                                              TensorArrayTypeFungible]> {
  let summary = "Array indexing operation";
  let description = [{
    Returns a single element from an array-typed operand.
  }];
  let arguments = (ins
    Xls_ArrayType:$array,
    // Note we only support a single index as Array is not multidimensional.
    Xls_Bits:$index
  );
  let results = (outs
    AnyType:$result
  );
  let builders = [
    OpBuilder<(ins "::mlir::Value":$array, "::mlir::Value":$index), [{
      ::Type element_type =
        ::cast<ArrayType>(array.getType()).getElementType();
      build($_builder, $_state, element_type, array, index);
    }]>
  ];
  let hasCanonicalizeMethod = 1;
}

def Xls_ArrayIndexStaticOp : Xls_Op<"array_index_static", [Pure, ShapesAreConsistent<["array", "result"]>,
                                                           TensorArrayTypeFungible]> {
  let summary = "Array indexing operation";
  let description = [{
    Returns a single element from an array-typed operand.
  }];
  let arguments = (ins
    Xls_ArrayType:$array,
    // Note we only support a single index as Array is not multidimensional.
    I64Attr:$index
  );
  let results = (outs
    AnyType:$result
  );
  let builders = [
    OpBuilder<(ins "::mlir::Value":$array, "int":$index), [{
      ::Type element_type =
        ::cast<ArrayType>(array.getType()).getElementType();
      build($_builder, $_state, element_type, array, index);
    }]>
  ];
  let hasCanonicalizeMethod = 1;
}

def Xls_ArraySliceOp : Xls_Op<"array_slice", [Pure, ShapesAreConsistent<["array", "start", "result"]>,
                                              TensorArrayTypeFungible]> {
  let summary = "Array slicing operation";
  let description = [{
    Returns a slice of an array.
  }];
  let arguments = (ins
    Xls_ArrayType:$array,
    Xls_Bits:$start,
    I64Attr:$width
  );
  let results = (outs
    Xls_ArrayType:$result
  );
}

def Xls_ArrayUpdateOp : Xls_Op<"array_update", [Pure, ShapesAreConsistent<["array", "value", "result"]>,
                                                TensorArrayTypeFungible]> {
  let summary = "Array updating operation";
  let description = [{
    Returns a modified copy of an array.
  }];
  let arguments = (ins
    Xls_ArrayType:$array,
    Xls_Bits:$value,
    Xls_Bits:$index
  );
  let results = (outs
    Xls_ArrayType:$result
  );
  let builders = [
    OpBuilder<(ins "::mlir::Value":$array, "::mlir::Value":$value, "::mlir::Value":$index), [{
      build($_builder, $_state, array.getType(), array, value, index);
    }]>
  ];

}

def Xls_ArrayUpdateSliceOp : Xls_Op<"array_update_slice", [Pure, ShapesAreConsistent<["array", "slice", "result"]>,
                                                            TensorArrayTypeFungible]> {
  let summary = "Array updating operation";
  let description = [{
    Returns an array with a slice updated. The slice is specified by a variable
    start index and a fixed width.
  }];
  let arguments = (ins
    Xls_ArrayType:$array,
    Xls_ArrayType:$slice,
    // TODO(jpienaar): Expand to support Xls_ScalarBits.
    I32:$start,
    I64Attr:$width
  );
  let results = (outs
    Xls_ArrayType:$result
  );
  let builders = [
    OpBuilder<(ins "::mlir::Value":$array, "::mlir::Value":$slice, "::mlir::Value":$index), [{
      int count = ::cast<ArrayType>(slice.getType()).getNumElements();
      build($_builder, $_state, array.getType(), array, slice, index, $_builder.getI64IntegerAttr(count));
    }]>
  ];
  // The custom<SameOperandsAndResultType> binds the array and result types.
  // The custom<ArrayUpdateSliceBrackets> binds the slice type.
  let assemblyFormat = [{
    $slice `into` $array `[` $start `+` `` `:` $width `]` attr-dict `:`
    custom<SameOperandsAndResultType>(type($array), type($result))
    custom<ArrayUpdateSliceBrackets>(ref(type($array)), ref($width), type($slice))
  }];
}

def Xls_ArrayConcatOp : Xls_Op<"array_concat", [Pure, TensorArrayTypeFungible]> {
  let summary = "Array concatenate";
  let description = [{
    Returns an array containing elements from all input arrays, concatenated.
  }];
  let arguments = (ins
    Variadic<Xls_ArrayType>:$arrays
  );
  let results = (outs
    Xls_ArrayType:$result
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Tuple operations
//===----------------------------------------------------------------------===//

def Xls_TupleOp : Xls_Op<"tuple", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                   TensorArrayTypeFungible]> {
  let summary = "Tuple operation";
  let description = [{
    Constructs a tuple of its operands.
  }];
  let arguments = (ins
    Variadic<AnyType>:$operands
  );
  let results = (outs
    TupleOf<[AnyType]>:$result
  );
}

def Xls_TupleIndexOp : Xls_Op<"tuple_index", [Pure, TensorArrayTypeFungible]> {
  let summary = "Tuple indexing operation";
  let description = [{
    Returns a single element from a tuple-typed operand.
  }];
  let arguments = (ins
    TupleOf<[AnyType]>:$operand,
    I64Attr:$index
  );
  let results = (outs
    AnyType:$result
  );
  let builders = [
    OpBuilder<(ins "::mlir::Value":$operand, "int":$index), [{
      build($_builder, $_state,
            ::cast<::mlir::TupleType>(operand.getType()).getType(index),
            operand, $_builder.getI64IntegerAttr(index));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Bit-vector operations
//===----------------------------------------------------------------------===//

def Xls_BitSliceOp : Xls_UnaryOp<"bit_slice",
    [Pure, ShapesAreConsistent<["operand", "result"]>]> {
  let summary = "Bit slice";
  let description = [{
    Slices a contiguous range of bits from a bits-typed operand.
  }];
  let arguments = (ins
    Xls_Bits:$operand,
    I64Attr:$start,
    I64Attr:$width
  );
  let results = (outs
    Xls_Bits:$result
  );
}

def Xls_BitSliceUpdateOp : Xls_Op<"bit_slice_update",
    [Pure, ShapesAreConsistent<["operand", "start", "update_value", "result"]>]> {
  let summary = "Bit slice update";
  let description = [{
    Replaces a contiguous range of bits in a bits-typed operand at a variable
    start index with a given value.
  }];
  let arguments = (ins
    Xls_Bits:$operand,
    Xls_Bits:$start,
    Xls_Bits:$update_value
  );
  let results = (outs
    Xls_Bits:$result
  );
}

def Xls_DynamicBitSliceOp : Xls_Op<"dynamic_bit_slice",
    [Pure, ShapesAreConsistent<["operand", "result"]>]> {
  let summary = "Dynamic bit slice";
  let description = [{
    Slices a contiguous range of bits from a bits-typed operand, with variable
    starting index but fixed width. Out-of-bounds slicing is supported by
    treating all out-of-bounds bits as having value 0.
  }];
  let arguments = (ins
    Xls_Bits:$operand,
    Xls_Bits:$start,
    I64Attr:$width
  );
  let results = (outs
    Xls_Bits:$result
  );
}

def Xls_ConcatOp : Xls_Op<"concat", [Pure]> {
  let summary = "Concatenate";
  let description = [{
    Concatenates and arbitrary number of bits-typed operands.
  }];
  let arguments = (ins
    Variadic<Xls_Bits>:$inputs
  );
  let results = (outs
    Xls_Bits:$result
  );
 let assemblyFormat = [{
    $inputs attr-dict `:` functional-type($inputs, $result)
  }];
}

def Xls_ReverseOp : Xls_UnaryOp<"reverse", [Pure, SameOperandsAndResultType]> {
  let summary = "Bit reversal";
  let description = [{
    Reverses the order of bits of its operand.
  }];
  let arguments = (ins
    Xls_Bits:$operand
  );
  let results = (outs
    Xls_Bits:$result
  );
}

def Xls_DecodeOp : Xls_UnaryOp<"decode", [Pure, ShapesAreConsistent<["operand", "result"]>]> {
  let summary = "Binary decode";
  let description = [{
    Implements a binary decoder.

    decode converts the binary-encoded operand value into a one-hot result. For
    an operand value of n interpreted as an unsigned number the n-th result bit
    and only the n-th result bit is set. The width of the decode operation may
    be less than the maximum value expressible by the input (2**N - 1). If the
    encoded operand value is larger than the number of bits of the result the
    result is zero.

    The result width M must be less than or equal to 2**N where N is the operand width.
  }];
  let arguments = (ins
    Xls_Bits:$operand,
    OptionalAttr<I64Attr>:$width
  );
  let results = (outs
    Xls_Bits:$result
  );
}

def Xls_EncodeOp : Xls_UnaryOp<"encode", [Pure, ShapesAreConsistent<["operand", "result"]>]> {
  let summary = "Binary encode";
  let description = [{
    Implements a binary encoder.

    The result width M must be equal to log2(N).

    encode converts the one-hot operand value into a binary-encoded value of the
    "hot" bit of the input. If the n-th bit and only the n-th bit of the operand
    is set the result is equal the value n as an unsigned number.

    If multiple bits of the input are set the result is equal to the logical or
    of the results produced by the input bits individually. For example, if bit
    3 and bit 5 of an encode input are set the result is equal to 3 | 5 = 7.

    If no bits of the input are set the result is zero.
  }];
  let arguments = (ins
    Xls_Bits:$operand
  );
  let results = (outs
    Xls_Bits:$result
  );
}

def Xls_OneHotOp : Xls_UnaryOp<"one_hot", [Pure, ShapesAreConsistent<["operand", "result"]>]> {
  let summary = "One-hot encoding";
  let description = [{
    Produces a bits value with exactly one bit set. The index of the set bit
    depends upon the input value.

    This operation can be thought of like a command: "canonicalize/normalize
    this input value to be one-hot", where lsb_prio indicates which side of the
    bitvector wins in the case that there are multiple input bits set. Contrast
    this with the decode operation which instead converts a binary-encoded
    operand value into a one-hot value.

    (Note: once this operation has been applied to a value, the optimizer knows
    that the output has the one-hot property [i.e. exactly one bit is set],
    which helps the optimizer know that one-hot-select operations that use the
    output value as a selector are selecting between distinct values.)
  }];
  let arguments = (ins
    Xls_Bits:$operand,
    BoolAttr:$lsb_prio
  );
  let results = (outs
    Xls_Bits:$result
  );
}

//===----------------------------------------------------------------------===//
// Control-oriented operations
//===----------------------------------------------------------------------===//

def Xls_SelOp : Xls_Op<"sel", [
    Pure,
    ShapesAreConsistent<["cases", "otherwise", "result"]>,
    AttrSizedOperandSegments,
    Elementwise,
    Scalarizable,
    TensorArrayTypeFungible]> {
  let summary = "Select";
  let description = [{
    Selects between operands based on a selector value.

    NOTE: The default case comes before the other cases!

    This behaves as if the selector indexes into the values given in cases,
    providing default if it is indexing beyond the given cases.
  }];
  let arguments = (ins
    Xls_Bits:$selector,
    Optional<Xls_BitsOrTuple>:$otherwise,  // Avoiding "default" as this is a C++ keyword.
    Variadic<Xls_BitsOrTuple>:$cases
  );
  let results = (outs
    Xls_BitsOrTuple:$result
  );
  let assemblyFormat = [{
    $selector `in` ` ` `[` $cases `]` (`else` $otherwise^)? attr-dict
    `:` `(` type($selector) `,` `[` type($cases) `]` (`,` type($otherwise)^)? `)` `->` type($result)
  }];
  let builders = [
    OpBuilder<(ins "::mlir::Value":$selector, "::mlir::Value":$otherwise, "::mlir::ValueRange":$cases), [{
      build($_builder, $_state, otherwise.getType(), selector, otherwise, cases);
    }]>,
    OpBuilder<(ins "::mlir::Type":$resultType, "::mlir::Value":$selector, "::mlir::ValueRange":$cases), [{
      build($_builder, $_state, resultType, selector, mlir::Value(), cases);
    }]>,
    OpBuilder<(ins "::mlir::Value":$selector, "::mlir::ValueRange":$cases), [{
      assert(!cases.empty() && "'cases' must be non-empty");
      build($_builder, $_state, cases.front().getType(), selector, mlir::Value(), cases);
    }]>,
    // This builder provides a "Select" like interface. The operand ordering to
    // actually implement this is backwards, so we accept a tag-struct that
    // unambiguously names the parameters to the user.
    OpBuilder<(ins "SelectBuilder":$tag), [{
      build($_builder, $_state, tag.getType(), tag.getSelector(), tag.getThen(), ::ValueRange{tag.getElse()});
    }]>
  ];
}

def Xls_OneHotSelOp : Xls_Op<"one_hot_sel", [
  Pure,
  ShapesAreConsistent<["cases", "result"]>,
  Elementwise,
  Scalarizable,
  TensorArrayTypeFungible]> {
  let summary = "One-hot select";
  let description = [{
    Selects between operands based on a one-hot selector, OR-ing all selected
    cases if more than one case is selected.

    See one_hot for an example of the one-hot selector invariant. Note that when
    the selector is not one-hot, this operation is still well defined.

    Note that when one_hot operations are used to precondition the selector
    operand to one_hot_sel, the XLS optimizer will try to determine when they
    are unnecessary and subsequently eliminate them.
  }];
  let arguments = (ins
    Xls_Bits:$selector,
    Variadic<Xls_Bits>:$cases
  );
  let results = (outs
    Xls_Bits:$result
  );
}

def Xls_PrioritySelOp : Xls_Op<"priority_sel", [
  Pure,
  ShapesAreConsistent<["cases", "default_value", "result"]>,
  Elementwise,
  Scalarizable,
  TensorArrayTypeFungible]> {
  let summary = "Priority select";
  let description = [{
    Selects between operands based on a selector, choosing the highest-priority
    case if more than one case is selected. Each bit in the selector corresponds
    to a case, with the least significant bit corresponding to the first case
    and having the highest priority. If there are no bits in the selector set,
    no case is selected and the default value of 0 is chosen.

    See one_hot for an example of the one-hot selector invariant. Note that when
    the selector is not one-hot, this operation is still well defined.

    Note that when one_hot operations are used to precondition the selector
    operand to priority_sel, the XLS optimizer will try to determine when they
    are unnecessary and subsequently eliminate them.
  }];
  let arguments = (ins
    Xls_Bits:$selector,
    Variadic<Xls_BitsOrTuple>:$cases,
    Xls_BitsOrTuple:$default_value
  );
  let results = (outs
    Xls_BitsOrTuple:$result
  );
  let assemblyFormat = [{
    $selector `in` ` ` `[` $cases `]` `else` $default_value attr-dict `:` `(` type($selector) `,` `[` type($cases) `]` `,` type($default_value) `)` `->` type($result)
  }];
}

// Invoke is just a func.CallOp.

def Xls_MapOp : Xls_Op<"map", []> {
  let summary = "Map a function over an array";
  let description = [{
    Applies a function to the elements of an array and returns the result as an
    array.
  }];
  let arguments = (ins
    Xls_ArrayType:$operand,
    FlatSymbolRefAttr:$to_apply
  );
  let results = (outs
    Xls_ArrayType:$result
  );
}

// TODO(jmolloy): DynamicCountedFor (but the XLS code generator cannot handle
// it).

def Xls_YieldOp : Xls_Op<"yield", [Pure, ReturnLike, Terminator]> {
  let summary = "for return op";
  let description = [{
    Yields an SSA value from the enclosing ForOp, EprocOp, or SprocOp region.
  }];

  let arguments = (ins Variadic<AnyType>:$results);

  let assemblyFormat = "attr-dict ($results^ `:` type($results))?";

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, ::ValueRange{});
    }]>
  ];
}

def Xls_ForOp : Xls_Op<"for", [
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"YieldOp">,
  Pure,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>]> {
 let summary = "Region-based fixed trip count loop";
 let description = [{
   for invokes the region trip_count times.

   The region is passed carry and invariant arguments. The region must yield
   carried values.

   This is an easier compiler target than CountedForOp and will be lowered into
   a CountedForOp.
 }];
 let arguments = (ins
    Variadic<Xls_BitsOrTuple>:$inits,
    Variadic<Xls_Bits>:$invariants,
    I64Attr:$trip_count
  );
  let results = (outs
    Variadic<Xls_BitsOrTuple>:$results
  );
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `inits` `(` $inits `)` `invariants` `(` $invariants `)` regions attr-dict `:` functional-type(operands, results)
  }];
}

def Xls_CountedForOp : Xls_Op<"counted_for",
    [CallOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Invokes a fixed-trip count loop";
  let description = [{
    counted_for invokes the function body trip_count times, passing loop-carried
    data that starts with value init.

    The first argument passed to body is the induction variable -- presently,
    the induction variable always starts at zero and increments by stride after
    every trip.

    The second argument passed to body is the loop-carry data. The return type
    of body must be the same as the type of the init loop carry data. The value
    returned from the last trip is the result of the counted_for expression.

    All subsequent arguments passed to body are passed from invariant_args; e.g.
    if there are two members in invariant_args those values are passed as the
    third and fourth arguments.

    Therefore body should have a signature that matches the following:

        body(i, loop_carry_data[, invariant_arg0, invariant_arg1, ...])

    Note that we currently inspect the body function to see what type of
    induction variable (i above) it accepts in order to pass an i value of that
    type.
  }];
  let arguments = (ins
    Xls_BitsOrTuple:$init,
    Variadic<Xls_Bits>:$invariant_args,
    I64Attr:$trip_count,
    FlatSymbolRefAttr:$to_apply,
    DefaultValuedOptionalAttr<I64Attr, "1">:$stride
  );
  let results = (outs
    Xls_BitsOrTuple:$result
  );

  let extraClassDeclaration = [{
    // Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
    mlir::MutableOperandRange getArgOperandsMutable() {
      return getInvariantArgsMutable();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    // Return the callee of this operation.
    mlir::CallInterfaceCallable getCallableForCallee() {
      return getToApplyAttr();
    }
    // Set the callee from the callable.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee);
  }];
}

//===----------------------------------------------------------------------===//
// Sequencing operations
//===----------------------------------------------------------------------===//

def Xls_AfterAllOp : Xls_Op<"after_all", [Pure]> {
  let summary = "Constructs partial orderings among channel operations";
  let description = [{
    Used to construct partial orderings among channel operations.

    after_all can consume an arbitrary number of token operands including zero.
  }];
  let arguments = (ins
    Variadic<Xls_TokenType>:$inputs
  );
  let results = (outs
    Xls_TokenType:$result
  );
  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, TokenType::get($_builder.getContext()), ::ValueRange{});
    }]>
  ];
  let assemblyFormat = [{
    $inputs attr-dict `:` custom<VariadicSameOperandsAndResultType>(ref($inputs), type($inputs), type($result))
  }];
}

//===----------------------------------------------------------------------===//
// Constant operation
//===----------------------------------------------------------------------===//

def Xls_ConstantTensorOp : Xls_Op<"constant_tensor", [ConstantLike, Pure]> {
  let summary = "Constant tensor operation";
  let description = [{
    Produces an output from a constant value.
  }];
  let arguments = (ins
    ElementsAttr:$value
  );
  let results = (outs
    StaticShapeTensorOf<[AnySignlessInteger, AnyFloat]>:$result
  );
  let hasFolder = 1;
}

def Xls_ConstantScalarOp : Xls_Op<"constant_scalar", [ConstantLike, Pure]> {
  let summary = "Constant scalar operation";
  let description = [{
    Produces an output from a constant value.
  }];
  let arguments = (ins
    AnyAttr:$value
  );
  let results = (outs
    Xls_ScalarBits:$result
  );
  let builders = [
    OpBuilder<(ins "::mlir::IntegerType":$type, "int32_t":$value), [{
      $_state.addTypes(type);
      $_state.addAttribute("value", $_builder.getI32IntegerAttr(value));
    }]>,
    OpBuilder<(ins "int32_t":$value), [{
      // Shorthand for I32.
      $_state.addTypes($_builder.getI32Type());
      $_state.addAttribute("value", $_builder.getI32IntegerAttr(value));
    }]>
  ];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// DSLX operations
//===----------------------------------------------------------------------===//

def Xls_ImportDslxFilePackageOp : Xls_Op<"import_dslx_file_package", [Symbol]> {
  let summary = "DSLX package import operation";
  let description = [{
    Represents importing a DSLX file package into the current package. The
    symbol name is used to refer to the imported package.

    This doesn't define which function to load or exposes symbols, instead it
    just represents a translate time loading of the DSLX module.  Template
    instantiations need to be handled inside the file itself.
  }];
  let arguments = (ins
    StrAttr:$filename,
    SymbolNameAttr:$sym_name
    // TODO(jpienaar): Expand to include search paths, define symbols etc.
  );
  let assemblyFormat = [{
    $filename `as` $sym_name attr-dict
  }];
  let extraClassDeclaration = [{
    // Imports can always be discarded if they are not used.
    bool canDiscardOnUseEmpty() const { return true; }
  }];
}

def Xls_CallDslxOp : Xls_Op<"call_dslx", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Call DSLX function";
  let description = [{
    This is an "unfactored" form of calling a DSLX function that is a combined
    package import and function call. The combined op is useful to enable
    inserting calls locally without needing to add the import at Module level.
    It is meant to be factored as part of lowering to XLS proper to reduce
    import overhead.
  }];
  let arguments = (ins
    StrAttr:$filename,
    StrAttr:$function,
    Variadic<AnyType>:$operands,
    UnitAttr:$is_pure
    // TODO(jpienaar): Expand to include search paths, define symbols etc.
  );
  let results = (outs AnyType);
  let assemblyFormat = [{
    $filename `:` $function `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

// TODO(jpienaar): Add variant that has string literal.

//===----------------------------------------------------------------------===//
// Elaborated procs
//===----------------------------------------------------------------------===//
//
// Elaborated procs are a form of lowered proc that has its channels "baked in"
// and cannot spawn new procs. Trees of procs can be lowered to eprocs.
def Xls_EprocOp : Xls_Op<"eproc", [
  Symbol,
  SingleBlockImplicitTerminator<"YieldOp">,
  IsolatedFromAbove,
  DeclareOpInterfaceMethods<XlsRegionOpInterface>,
  // Callable as it contains calls so should participate in the CallGraph.
  CallableOpInterface
]> {
  let summary = "Elaborated proc";
  let description = [{
    This is a recurrent function with a constant initial state.

    The function is called with one or more arguments, the current state. The
    function returns the new state. The number of results must equal the number
    of arguments. The function runs indefinitely.

    An eproc may communicate with the outside world via channels. Channels
    are declared within eprocs and are referenced by symbol name.

    Elaborated procs differ to Procs in that they have their channels "baked in"
    and cannot spawn new procs. Trees of procs can be lowered to eprocs.

    The initial state is always the all-zeroes constant, denoted using the
    keyword "zeroinitializer". This makes the initial state explicit to the
    reader as well as enabling extension to more complex initial states in the
    future without complicating the grammar.

    If `discardable` is true, the eproc may be discarded if it is not used.

    The `min_pipeline_stages` attribute has the same meaning as in SprocOp. See
    SprocOp for more details.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    UnitAttr:$discardable,
    DefaultValuedAttr<I64Attr, "1">:$min_pipeline_stages
  );
  let regions = (region
    SizedRegion<1>:$body
  );
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::Block::BlockArgListType getStateArguments() {
      return getBody().getArguments();
    }
    ::mlir::ValueRange getYieldedArguments() {
      return getBody().getBlocks().front().getTerminator()->getOperands();
    }
    mlir::Region* getCallableRegion() {
      return &getBody();
    }
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() {
      return {};
    }
    llvm::ArrayRef<::mlir::Type> getResultTypes() {
      return {};
    }
    bool canDiscardOnUseEmpty() { return getDiscardable(); }
  }];
}

def Xls_ChanOp : Xls_Op<"chan", [
  Symbol,
  // While a channel isn't theoretically "callable", it is referred to by sends
  // an receives in enough of a "call-like" way to participate in the CallGraph.
  CallableOpInterface]> {
  let summary = "Channel";
  let description = [{
    A channel is a unidirectional FIFO queue of values. Channels are declared
    at global scope and are referenced by symbol name. Channels carry values of
    a single type.

    Channels are named symbols so that they can be referenced by external procs.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type,
    DefaultValuedAttr<BoolAttr, "true">:$send_supported,
    DefaultValuedAttr<BoolAttr, "true">:$recv_supported
  );
  let assemblyFormat = [{
    $sym_name attr-dict `:` $type
  }];
  let extraClassDeclaration = [{
    // Channels can always be discarded if they are not used.
    bool canDiscardOnUseEmpty() const { return true; }

    mlir::Region* getCallableRegion() {
      return nullptr;
    }
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() {
      return {};
    }
    llvm::ArrayRef<::mlir::Type> getResultTypes() {
      return {};
    }
  }];

}

def Xls_InstantiateEprocOp : Xls_Op<"instantiate_eproc", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Instantiates an eproc";
  let description = [{
    Instantiates an eproc, binding the channels the eproc sends and receives on
    to the given "global" channels.

    "global" channels are channels that will be bound to the eproc's channels.
    "local" channels are those channels that the eproc already uses.

    During instantiation, the eproc will be cloned, and references to the local
    channels will be replaced with references to the global channels. The
    original eproc will be removed after instantiation.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$eproc,
    FlatSymbolRefArrayAttr:$global_channels,
    FlatSymbolRefArrayAttr:$local_channels
  );
  let assemblyFormat = [{
    $eproc custom<ZippedSymbols>($global_channels, $local_channels) attr-dict
  }];
}

def Xls_InstantiateExternEprocOp : Xls_Op<"instantiate_extern_eproc", []> {
  let summary = "Binds an externally defined eproc";
  let description = [{
    Binds channels to an eproc that is defined externally to this module.

    This functions similarly to the `xls.instantiate_eproc` op, except that:
      1) The eproc definition is not available in this module and so it is
         referred to by string, not a symbol.
      2) Instead of binding global channels to local channels, it binds global
         channels to the boundary channel names of the target eproc.

    The target eproc is referred to by an opaque string. The interpretation of
    this string is left to the user.
  }];
  let arguments = (ins
    StrAttr:$eproc_name,
    FlatSymbolRefArrayAttr:$global_channels,
    StrArrayAttr:$boundary_channel_names
  );
  let assemblyFormat = [{
    $eproc_name custom<ZippedSymbols>($global_channels, $boundary_channel_names) attr-dict
  }];
}

def Xls_NextValueOp : Xls_Op<"next_value", [Pure, SameVariadicOperandSize]> {
  let summary = "Selects the next value to yield for eproc";
  let description = [{
    The next_value op represents, if feeding into a yield of a elaborated proc,
    the selected next value to yield. It consists of pairs of values, where the
    first value is the predicate and the second value is the value to yield if
    the predicate is true. The operation assumes that the predicates are
    disjoint at runtime (only one may be true).

    This opereation has no semantics and can be deleted if it does not feed
    directly into a xls.yield op.
  }];
  let arguments = (ins
    Variadic<I1>:$predicates,
    Variadic<Xls_Bits>:$values
  );
  let results = (outs
    Xls_Bits:$result
  );
  // TODO(jpienaar): The verifier should be made stricter (check element type
  // matches).
  let assemblyFormat = [{
    custom<NextValuePair>($predicates, $values) attr-dict `:` functional-type($values, results)
  }];
}

//===----------------------------------------------------------------------===//
// Structured procs
//===----------------------------------------------------------------------===//

def Xls_SchanType : Xls_Type<"Schan", []> {
  let summary = "structured channel";
  let mnemonic = "schan";
  let parameters = (ins
    TypeParameter<"::mlir::Type", "Underlying type">:$element_type,
    TypeParameter<"bool", "Direction is input?">:$is_input
  );
  let assemblyFormat = "`<` $element_type `,` custom<InOutSpecifier>($is_input) `>`";
  let extraClassDeclaration = [{
    bool getIsOutput() const { return !getIsInput(); }
  }];
}

def Xls_SchanOp : Xls_Op<"schan", [Pure, HasParent<"SprocOp">]> {
  let summary = "structured channel declaration";
  let description = [{
    Declares a structured channel. Returns the out and in channels. The first
    result is the out (sender) channel, the second is the in (receiver) channel.
  }];
  let arguments = (ins
    StrAttr:$name,
    TypeAttr:$type
  );
  let results = (outs
    Xls_SchanType:$out,
    Xls_SchanType:$in
  );
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "::mlir::StringRef":$name, "::mlir::Type":$element_type), [{
      auto inChanTy = SchanType::get($_builder.getContext(), element_type, /*isInput=*/true);
      auto outChanTy = SchanType::get($_builder.getContext(), element_type, /*isInput=*/false);
      build($_builder, $_state, ::mlir::TypeRange{outChanTy, inChanTy}, name, element_type);
    }]>
  ];
}

def Xls_SprocOp : Xls_Op<"sproc", [
  Symbol,
  SingleBlockImplicitTerminator<"YieldOp">,
  IsolatedFromAbove,
  CallableOpInterface,
  DeclareOpInterfaceMethods<XlsRegionOpInterface>,
]> {
  let summary = "Structured proc";
  let description = [{
    Structured procs are a high-level form of proc that can be lowered to eprocs.
    Eprocs resemble the XLS-IR proc representation, where Sprocs resemble the
    DSLX proc representation.

    Sprocs look like functions that take structured channels, Schans.
    Schans are unidirectional typed FIFOs with their direction baked in.

    Within the Sproc are two regions, the "spawns" and "next" regions. The spawns
    region contains Schan declarations and "spawns" of other Sprocs. The only way
    to create an Sproc is to spawn another Sproc.

    Sprocs may be marked as "top" if they EITHER:
      a) Have no channels.
      b) Have the optional `boundary_channel_names` attribute set.

    In the latter case, boundary channels (used by XLS for tests etc) are
    created with the given names.

    The argument channels are available within the spawns region, and new
    channels may be declared within the spawns region. The spawns region yields
    channels to the next region; only channels yielded are available within the
    next region.

    Note that argument channels may be directly yielded and hence become
    available to the next region.

    The next region is the main body of the proc. It takes the channels yielded
    from the spawns region plus state arguments. It must yield the next state
    arguments.

    ```
    xls.sproc @myproc (%arg0: !xls.schan<i32, in>) {
      spawns {
        %mychan_in, %mychan_out = xls.schan<i32>("mychan")
        xls.spawn @myproc_child(%arg0, %mychan_out) : !xls.schan<i32, in>, !xls.schan<i32, out>
        xls.yield %mychan_in : !xls.schan<i32, in>
      }
      next (%chan0: !xls.schan<i32, in>, %state: i32) zeroinitializer {
        %0 = xls.srecv %chan0 : i32
        %1 = xls.add %0, %state : i32
        xls.yield %1 : i32
      }
    }
    ```

    The `min_pipeline_stages` attribute tracks the mininum number of pipeline
    stages that XLS must be given in order to have a chance of scheduling the
    proc. The default is 1, but some transforms guarantee that the proc will no
    longer schedule in one pipeline stage; for example:

      myop(...) -> xls.sblocking_receive(xls.ssend(...))

    The above transform will guarantee that the proc requires at least two
    pipeline stages.

    The min_pipeline_stages attribute is not intended to guarantee that the proc
    will schedule in the given number of pipeline stages, only that it is known
    that it will *not* schedule in fewer than the given number of pipeline
    stages.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    BoolAttr:$is_top,
    OptionalAttr<ArrayAttr>:$boundary_channel_names,
    DefaultValuedAttr<I64Attr, "1">:$min_pipeline_stages
  );
  let regions = (region
    SizedRegion<1>:$spawns,
    SizedRegion<1>:$next
  );
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::ArrayRef<::mlir::BlockArgument> getStateArguments() {
      ::mlir::ArrayRef<::mlir::BlockArgument> args =
          getNext().getArguments();
      return args.drop_while([](::mlir::BlockArgument arg) {
        return ::mlir::isa<SchanType>(arg.getType());
      });
    }
    ::mlir::ValueRange getChannelArguments() {
      ::mlir::ArrayRef<::mlir::BlockArgument> args =
          getSpawns().getArguments();
      return args;
    }
    ::mlir::ValueRange getNextChannels() {
      ::mlir::ArrayRef<::mlir::BlockArgument> args =
          getNext().getArguments();
      return args.take_while([](::mlir::BlockArgument arg) {
        return ::mlir::isa<SchanType>(arg.getType());
      });
    }
    ::mlir::ValueRange getYieldedChannels() {
      if (getSpawns().empty() ||
          !getSpawns().getBlocks().front().mightHaveTerminator()) {
        return {};
      }
      return getSpawns().getBlocks().front().getTerminator()->getOperands();
    }
    ::mlir::ValueRange getYieldedArguments() {
      return getNext().getBlocks().front().getTerminator()->getOperands();
    }

    ::mlir::Region* getCallableRegion() {
      // Spawns is where the calls (spawns) are.
      return &getSpawns();
    }
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() {
      return {};
    }
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() {
      return {};
    }

    // Adds the given values to the yield op and the next region. Returns the
    // index of the first value in both the yield operands and the
    // getNextChannels().
    int64_t addYieldedChannels(ValueRange values) {
      Operation* yield = getSpawns().getBlocks().front().getTerminator();
      int64_t index = getNext().getNumArguments() - getStateArguments().size();
      yield->insertOperands(index, values);
      int64_t i = index;
      for (Value value : values) {
        getNext().insertArgument(i++, value.getType(), value.getLoc());
      }
      return index;
    }

    ::mlir::TypeRange getChannelArgumentTypes() {
      return getChannelArguments().getTypes();
    }

    bool canDiscardOnUseEmpty() { return !getIsTop(); }
  }];
}

def Xls_SpawnOp : Xls_Op<"spawn", [
  HasParent<"SprocOp">,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  CallOpInterface]> {
  let summary = "spawn sproc";
  let description = [{
    Spawns an sproc.
  }];
  let arguments = (ins
    Variadic<Xls_SchanType>:$channels,
    SymbolRefAttr:$callee
  );
  let assemblyFormat = [{
    $callee `(` operands `)` attr-dict `:` type(operands)
  }];
  let extraClassDeclaration = [{
    SprocOp resolveCallee(::mlir::SymbolTableCollection* symbolTable = nullptr);
    ExternSprocOp resolveExternCallee(::mlir::SymbolTableCollection* symbolTable = nullptr);

    ::mlir::CallInterfaceCallable getCallableForCallee() {
      return getCallee();
    }
    void setCalleeFromCallable(::mlir::CallInterfaceCallable callable) {
      llvm_unreachable("setCalleeFromCallable is not supported");
    }
    ::mlir::Operation::operand_range getArgOperands() {
      return getChannels();
    }
    ::mlir::MutableOperandRange getArgOperandsMutable() {
      return getChannelsMutable();
    }
  }];
}

def Xls_ExternSprocOp : Xls_Op<"extern_sproc", [
  Symbol,
  CallableOpInterface
]> {
  let summary = "extern sproc";
  let description = [{
    Declares an sproc that is external to the current module. The sproc is
    spawnable by sprocs in the current module.

    The `boundary_channel_names` attribute is used to name each argument or
    result channel. These correspond to the `boundary_channel_names` on the
    target sproc (wherever it is defined). The `channel_types` attribute is used
    to specify the types of the channels.

    An `spawn` of a `extern_sproc` is lowered to an `instantiate_extern_eproc`
    op.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrArrayAttr:$boundary_channel_names,
    TypeArrayAttr:$channel_argument_types
  );
  let assemblyFormat = [{
    $sym_name custom<ChannelNamesAndTypes>($boundary_channel_names, $channel_argument_types) attr-dict
  }];
  let extraClassDeclaration = [{
    ::mlir::Region* getCallableRegion() {
      return nullptr;
    }
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() {
      return {};
    }
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() {
      return {};
    }
    bool canDiscardOnUseEmpty() { return true; }
  }];
}

def Xls_SBlockingReceiveOp : Xls_Op<"sblocking_receive", []> {
  let summary = "structured blocking receive";
  let description = [{
    Receives a data value from a specified channel. The type of the data value
    is determined by the channel. An optional predicate value conditionally
    enables the receive operation. A blocking receive waits (or blocks) until
    valid data is present at the channel.

    If the predicate is false the data values in the result are zero-filled.
  }];
  let arguments = (ins
    Xls_TokenType:$tkn,
    Xls_SchanType:$channel,
    Optional<I1>:$predicate
  );
  let results = (outs
    Xls_TokenType:$tkn_out,
    Xls_BitsOrTuple:$result
  );
  let builders = [
    OpBuilder<(ins "::mlir::Value":$token, "::mlir::Value":$channel), [{
      Type element_type = cast<SchanType>(channel.getType()).getElementType();
      build($_builder, $_state, token.getType(), element_type, token, channel, nullptr);
    }]>,
    OpBuilder<(ins "::mlir::Value":$channel), [{
      ::mlir::Value token = $_builder.create<AfterAllOp>($_builder.getUnknownLoc());
      Type element_type = cast<SchanType>(channel.getType()).getElementType();
      build($_builder, $_state, token.getType(), element_type, token, channel, nullptr);
    }]>
  ];
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def Xls_SNonblockingReceiveOp : Xls_Op<"snonblocking_receive", []> {
  let summary = "structured nonblocking receive";
  let description = [{
    Receives a data value from a specified channel. The type of the data value
    is determined by the channel. An optional predicate value conditionally
    enables the receive operation. Compared to a blocking receive, a
    non-blocking receive has an additional entry in its return tuple of type
    bits[1] denoting whether the data read is valid.

    If the predicate is false the data values in the result are zero-filled.
  }];
  let arguments = (ins
    Xls_TokenType:$tkn,
    Xls_SchanType:$channel,
    Optional<I1>:$predicate
  );
  let results = (outs
    Xls_TokenType:$tkn_out,
    Xls_BitsOrTuple:$result,
    I1:$valid
  );
  let builders = [
    OpBuilder<(ins "::mlir::Value":$token, "::mlir::Value":$channel), [{
      Type element_type = cast<SchanType>(channel.getType()).getElementType();
      build($_builder, $_state, token.getType(), element_type, $_builder.getI1Type(),  token, channel, nullptr);
    }]>,
    OpBuilder<(ins "::mlir::Value":$channel), [{
      ::mlir::Value token = $_builder.create<AfterAllOp>($_builder.getUnknownLoc());
      Type element_type = cast<SchanType>(channel.getType()).getElementType();
      build($_builder, $_state, token.getType(), element_type, $_builder.getI1Type(), token, channel, nullptr);
    }]>
  ];
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def Xls_SSendOp : Xls_Op<"ssend", []> {
  let summary = "structured send";
  let description = [{
    Sends data to a specified channel. The type of the data values is determined
    by the channel. An optional predicate value conditionally enables the send
    operation.

    The type of data must match the type supported by the channel.
  }];
  let arguments = (ins
    Xls_TokenType:$tkn,
    Xls_BitsOrTuple:$data,
    Xls_SchanType:$channel,
    Optional<I1>:$predicate
  );
  let results = (outs
    Xls_TokenType:$result
  );
  let builders = [
    OpBuilder<(ins "::mlir::Value":$token, "::mlir::Value":$value, "::mlir::Value":$channel), [{
      build($_builder, $_state, token.getType(), token, value, channel, nullptr);
    }]>,
    OpBuilder<(ins "::mlir::Value":$value, "::mlir::Value":$channel), [{
      ::mlir::Value token = $_builder.create<AfterAllOp>($_builder.getUnknownLoc());
      build($_builder, $_state, token.getType(), token, value, channel, nullptr);
    }]>
  ];

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def Xls_TraceOp : Xls_Op<"trace", [TensorArrayTypeFungible,
                                   AttrSizedOperandSegments,
                                   PredicatableOpInterface]> {
  let summary = "trace";
  let description = [{
    Emits a trace event. This is an undocumented XLS node.

    It takes a format string and a list of arguments. The format string uses
    `{}` placeholders for the arguments.

    The verbosity is a number that is used to filter trace events.
  }];
  let arguments = (ins
    Xls_TokenType:$tkn,
    Optional<I1>:$predicate,
    Variadic<Xls_BitsOrTuple>:$args,
    StrAttr:$format,
    DefaultValuedAttr<I64Attr, "0">:$verbosity
  );
  let results = (outs
    Xls_TokenType:$tkn_out
  );
  let assemblyFormat = [{
    $tkn `,` ($predicate^ `,`)? $format (`(` $args^ `)` `:` type($args))? (`verbosity` $verbosity^)? attr-dict
  }];
  let extraClassDeclaration = [{
    ::mlir::Value getCondition() {
      return getPredicate();
    }
    void setCondition(::mlir::Value condition) {
      getPredicateMutable().assign(condition);
    }
  }];
  let builders = [
    OpBuilder<(ins "::mlir::Value":$token, "::mlir::Value":$predicate,
                   "::llvm::StringRef":$format,
                    CArg<"::mlir::ValueRange", "{}">:$args,
                    CArg<"int", "0">:$verbosity), [{
      build($_builder, $_state, token, predicate, args, $_builder.getStringAttr(format), $_builder.getI64IntegerAttr(verbosity));
    }]>,
    OpBuilder<(ins "::mlir::Value":$token,
                    "::llvm::StringRef":$format,
                    CArg<"::mlir::ValueRange", "{}">:$args,
                    CArg<"int", "0">:$verbosity), [{
      build($_builder, $_state, token, nullptr, args, $_builder.getStringAttr(format), $_builder.getI64IntegerAttr(verbosity));
    }]>,
    OpBuilder<(ins "::llvm::StringRef":$format,
                    CArg<"::mlir::ValueRange", "{}">:$args,
                    CArg<"int", "0">:$verbosity), [{
      ::mlir::Value token = $_builder.create<AfterAllOp>($_builder.getUnknownLoc());
      build($_builder, $_state, token, nullptr, args, $_builder.getStringAttr(format), $_builder.getI64IntegerAttr(verbosity));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Lowering internal operations
//===----------------------------------------------------------------------===//

def VectorizedCallOp : Xls_Op<"vectorized_call",
    [CallOpInterface,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "vectorized call operation";
  let description = [{
    Represents a call to a function taking scalar types and returning scalar
    types, but where the call is vectorized. The types of the callee are scalar
    but the types of the operands and results of this operation are vectorized.

    Example:

    ```mlir
    %0 = xls.vectorized_call @callee(%0, %1) :
      (tensor<...>, tensor<...>) -> tensor<...>
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<Xls_Bits>:$operands);
  let results = (outs Variadic<Xls_Bits>);

  let builders = [
     OpBuilder<(ins "::mlir::func::FuncOp":$callee,
                    CArg<"::mlir::ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(vectorizeTypesAs(callee.getFunctionType().getResults(),
                                        operands.front()));
    }]>,
  ];

  let extraClassDeclaration = [{
    // Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
    mlir::MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    // Return the callee of this operation.
    mlir::CallInterfaceCallable getCallableForCallee() {
      return getCalleeAttr();
    }
    // Set the callee from the callable.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee);
  }];

  let assemblyFormat = [{
    $callee `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}


#endif
