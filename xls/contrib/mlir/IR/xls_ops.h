// Copyright 2024 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef GDM_HW_MLIR_XLS_IR_XLS_OPS_H_
#define GDM_HW_MLIR_XLS_IR_XLS_OPS_H_

#include "llvm/include/llvm/ADT/StringRef.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/include/mlir/IR/Attributes.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/Dialect.h"
#include "mlir/include/mlir/IR/DialectImplementation.h"
#include "mlir/include/mlir/IR/Location.h"
#include "mlir/include/mlir/IR/MLIRContext.h"
#include "mlir/include/mlir/IR/OpDefinition.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/TensorEncoding.h"
#include "mlir/include/mlir/IR/TypeUtilities.h"
#include "mlir/include/mlir/IR/Types.h"
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/include/mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/include/mlir/Transforms/DialectConversion.h"

// Include order below matters.
#include "mlir/include/mlir/IR/Value.h"
#include "xls/contrib/mlir/IR/xls_ops_enums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "xls/contrib/mlir/IR/xls_ops_attrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "xls/contrib/mlir/IR/xls_ops_typedefs.h.inc"

namespace mlir::xls {

class XlsDialect : public Dialect {
 public:
  explicit XlsDialect(mlir::MLIRContext* ctx);

  static llvm::StringRef getDialectNamespace() { return "xls"; }

  Attribute parseAttribute(DialectAsmParser& parser, Type type) const override;
  void printAttribute(Attribute attribute,
                      DialectAsmPrinter& printer) const override;
  Type parseType(DialectAsmParser& parser) const override;
  void printType(Type type, DialectAsmPrinter& printer) const override;
  Operation* materializeConstant(OpBuilder& builder, Attribute value, Type type,
                                 Location loc) override;
};

// An operation that can be applied to either tensors or arrays. Used for
// packing/unpacking operations like ArrayOp and TupleIndexOp that do not care
// about the nested type of their operands.
template <typename ConcreteType>
class TensorArrayTypeFungible
    : public OpTrait::TraitBase<ConcreteType, TensorArrayTypeFungible> {};

// Builder for a binary select operation. Can be passed to the builder of a
// SelOp.
class SelectBuilder {
 public:
  explicit SelectBuilder(Value selector) : selector(selector) {}
  SelectBuilder& Then(Value value) {
    then = value;
    return *this;
  }
  SelectBuilder& Else(Value value) {
    otherwise = value;
    return *this;
  }

  Type getType() const { return then.getType(); }
  Value getSelector() const { return selector; }
  Value getThen() const { return then; }
  Value getElse() const { return otherwise; }

 private:
  Value selector;
  Value then;
  Value otherwise;
};

}  // namespace mlir::xls

#define GET_INTERFACE_CLASSES
#include "xls/contrib/mlir/IR/interfaces.h.inc"  // IWYU pragma: export

#define GET_OP_CLASSES
#include "xls/contrib/mlir/IR/xls_ops.h.inc"  // IWYU pragma: export

#endif  // GDM_HW_MLIR_XLS_IR_XLS_OPS_H_
