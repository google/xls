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

#include "xls/contrib/mlir/tools/xls_translate/xls_translate_from_mlir.h"

#include <cassert>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/ADT/APFloat.h"
#include "llvm/include/llvm/ADT/APInt.h"
#include "llvm/include/llvm/ADT/STLExtras.h"
#include "llvm/include/llvm/ADT/Sequence.h"
#include "llvm/include/llvm/ADT/StringExtras.h"
#include "llvm/include/llvm/ADT/StringMap.h"
#include "llvm/include/llvm/ADT/StringRef.h"
#include "llvm/include/llvm/ADT/Twine.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"
#include "llvm/include/llvm/Support/Casting.h"  // IWYU pragma: keep
#include "llvm/include/llvm/Support/LogicalResult.h"
#include "llvm/include/llvm/Support/MemoryBuffer.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/Diagnostics.h"
#include "mlir/include/mlir/IR/Location.h"
#include "mlir/include/mlir/IR/SymbolTable.h"
#include "mlir/include/mlir/IR/TypeUtilities.h"
#include "mlir/include/mlir/IR/Types.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Pass/PassManager.h"
#include "mlir/include/mlir/Support/DebugStringHelper.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/Transforms/Passes.h"
#include "xls/codegen/vast/vast.h"
#include "xls/codegen/xls_metrics.pb.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/contrib/mlir/IR/xls_ops.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/foreign_function.h"
#include "xls/ir/foreign_function_data.pb.h"
#include "xls/ir/nodes.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/public/function_builder.h"
#include "xls/public/ir.h"
#include "xls/public/ir_parser.h"
#include "xls/public/runtime_build_actions.h"
#include "xls/tools/opt.h"

namespace mlir::xls {
namespace {

using ::llvm::zip;
using ::mlir::func::FuncOp;
using ::xls::BuilderBase;
using ::xls::BValue;
using ::xls::FunctionBuilder;
using ::xls::Package;
using ::xls::ProcBuilder;

namespace arith = ::mlir::arith;

using ValueMap = DenseMap<Value, BValue>;

// The default name of the result for imported Verilog module.
constexpr StringLiteral kResultName("out");

// The package name for imported DSLX instantiations.
constexpr StringLiteral kImportedModuleName("imported_module");

// Tracks imported package and the renaming due to import.
struct PackageInfo {
  PackageInfo(std::shared_ptr<const Package> p, Package::PackageMergeResult m,
              ImportDslxFilePackageOp op)
      : package(std::move(p)), merge_result(std::move(m)), op(op) {}

  std::shared_ptr<const Package> package;
  Package::PackageMergeResult merge_result;
  ImportDslxFilePackageOp op;
};

class TranslationState {
 public:
  explicit TranslationState(Package& package) : package_(package) {}

  void setValueMap(DenseMap<Value, BValue>& value_map) {
    this->value_map_ = &value_map;
  }

  BValue getXlsValue(Value value) const {
    auto it = value_map_->find(value);
    if (it == value_map_->end()) {
      llvm::errs() << "Failed to find value: " << value << "\n";
      return BValue();
    }
    return it->second;
  }

  void setValue(Value value, BValue xls_value) {
    value_map_->insert({value, xls_value});
  }

  void setMultiResultValue(Operation* op, BValue value, BuilderBase& fb) {
    for (auto [i, result] : llvm::enumerate(op->getResults())) {
      BValue extract = fb.TupleIndex(value, i, value.loc());
      value_map_->insert({result, extract});
    }
  }

  const Package& getPackage() const { return package_; }
  Package& getPackage() { return package_; }

  void addFunction(StringRef name, ::xls::Function* func) {
    func_map_[name] = func;
  }

  void addChannel(ChanOp op, ::xls::Channel* channel) {
    channel_map_[op] = channel;
  }

  ::xls::Channel* getChannel(ChanOp op) const {
    auto it = channel_map_.find(op);
    return it == channel_map_.end() ? nullptr : it->second;
  }

  const PackageInfo* getPackageInfo(StringRef name) const {
    auto it = package_map_.find(name);
    return it == package_map_.end() ? nullptr : &it->second;
  }

  void setPackageInfo(StringRef name, PackageInfo&& info) {
    package_map_.try_emplace(name, std::move(info));
  }

  void addLinkage(StringRef name, TranslationLinkage linkage) {
    linkage_map_[name] = linkage;
  }

  const TranslationLinkage* getLinkage(StringRef name) const {
    auto it = linkage_map_.find(name);
    return it == linkage_map_.end() ? nullptr : &it->second;
  }

  ::xls::SourceInfo getLoc(Operation* op) const;

  ::xls::Type* getType(Type t) const;

  LogicalResult recordOpaqueTypes(FuncOp func, ::xls::Function* xls_func);

  ::xls::Function* getFunction(StringRef name) const {
    auto it = func_map_.find(name);
    return it == func_map_.end() ? nullptr : it->second;
  }

  SymbolTableCollection& getSymbolTable() const { return symbol_table_; }

 private:
  DenseMap<Value, BValue>* value_map_;
  DenseMap<ChanOp, ::xls::Channel*> channel_map_;
  llvm::StringMap<::xls::Function*> func_map_;
  llvm::StringMap<PackageInfo> package_map_;
  llvm::StringMap<TranslationLinkage> linkage_map_;
  Package& package_;
  mutable llvm::DenseMap<Type, ::xls::Type*> type_map_;

  // Acts as a cache for symbol lookups, should be available even in const
  // functions.
  mutable SymbolTableCollection symbol_table_;
};

::xls::SourceInfo TranslationState::getLoc(Operation* op) const {
  Location loc = op->getLoc();
  if (isa<UnknownLoc>(loc)) {
    return ::xls::SourceInfo();
  }
  auto file_loc = dyn_cast<FileLineColLoc>(loc);
  if (!file_loc) {
    return ::xls::SourceInfo();
  }

  std::string_view filename(file_loc.getFilename().getValue());

  return ::xls::SourceInfo(
      package_.AddSourceLocation(filename, ::xls::Lineno(file_loc.getLine()),
                                 ::xls::Colno(file_loc.getColumn())));
}

::xls::Type* TranslationState::getType(Type t) const {
  auto& xls_type = type_map_[t];
  if (xls_type != nullptr) {
    return xls_type;
  }

  xls_type =
      TypeSwitch<Type, ::xls::Type*>(t)
          .Case<IntegerType>(
              [&](IntegerType t) { return package_.GetBitsType(t.getWidth()); })
          .Case<IndexType>(
              [&](IndexType /*t*/) { return package_.GetBitsType(32); })
          .Case<FloatType>([&](FloatType t) -> ::xls::Type* {
            // Note that getFPMantissaWidth() includes the sign bit.
            int exponent_width = t.getWidth() - t.getFPMantissaWidth();
            return package_.GetTupleType(
                {package_.GetBitsType(1), package_.GetBitsType(exponent_width),
                 package_.GetBitsType(t.getFPMantissaWidth() - 1)});
          })
          .Case<ArrayType>([&](ArrayType t) -> ::xls::Type* {
            return package_.GetArrayType(t.getNumElements(),
                                         getType(t.getElementType()));
          })
          .Case<TokenType>([&](TokenType /*t*/) -> ::xls::Type* {
            return package_.GetTokenType();
          })
          .Case<TupleType>([&](TupleType t) -> ::xls::Type* {
            auto range = llvm::map_range(t.getTypes(),
                                         [&](Type t) { return getType(t); });
            std::vector<::xls::Type*> types(range.begin(), range.end());
            return package_.GetTupleType(types);
          })
          .Case<OpaqueType>(
              [&](OpaqueType t) -> ::xls::Type* {
                llvm::errs()
                    << "Opaque type was not resolved during function imports: "
                    << t << "\n";
                return nullptr;
              })
          .Default([&](Type t) {
            llvm::errs() << "Unsupported type: " << t << "\n";
            return nullptr;
          });
  return xls_type;
}

LogicalResult TranslationState::recordOpaqueTypes(FuncOp func,
                                                  ::xls::Function* xls_func) {
  ::xls::FunctionType* xls_func_type = xls_func->GetType();
  for (auto [mlirType, xls_type] :
       zip(func.getArgumentTypes(), xls_func_type->parameters())) {
    if (isa<OpaqueType>(mlirType)) {
      type_map_[mlirType] = xls_type;
    }
  }
  std::vector<::xls::Type*> return_type_vec{xls_func_type->return_type()};
  absl::Span<::xls::Type* const> return_types(return_type_vec);
  if (func.getResultTypes().size() > 1) {
    CHECK(xls_func_type->return_type()->IsTuple());
    return_types =
        xls_func_type->return_type()->AsTupleOrDie()->element_types();
  }
  for (auto [mlirType, xls_type] : zip(func.getResultTypes(), return_types)) {
    if (isa<OpaqueType>(mlirType)) {
      type_map_[mlirType] = xls_type;
    }
  }
  return success();
}

// Unary bitwise ops.
#define XLS_UNARY_OP(TYPE, BUILDER)                                           \
  BValue convertOp(TYPE op, const TranslationState& state, BuilderBase& fb) { \
    return fb.BUILDER(state.getXlsValue(op.getOperand()), state.getLoc(op));  \
  }
XLS_UNARY_OP(IdentityOp, Identity);
XLS_UNARY_OP(NotOp, Not);

// Variadic bitwise operations
#define XLS_VARIADIC_BINARY_OP(TYPE, BUILDER)                                 \
  BValue convertOp(TYPE op, const TranslationState& state, BuilderBase& fb) { \
    std::vector<BValue> args;                                                 \
    for (auto v : op.getInputs()) args.push_back(state.getXlsValue(v));       \
    return fb.BUILDER(args, state.getLoc(op));                                \
  }
XLS_VARIADIC_BINARY_OP(AndOp, And);
XLS_VARIADIC_BINARY_OP(NandOp, Nand);
XLS_VARIADIC_BINARY_OP(OrOp, Or);
XLS_VARIADIC_BINARY_OP(NorOp, Nor);
XLS_VARIADIC_BINARY_OP(XorOp, Xor);

// Bitwise reduction operations
XLS_UNARY_OP(AndReductionOp, AndReduce);
XLS_UNARY_OP(OrReductionOp, OrReduce);
XLS_UNARY_OP(XorReductionOp, XorReduce);

// Arithmetic unary operations
XLS_UNARY_OP(NegOp, Negate);

// Binary ops.
#define XLS_BINARY_OP(TYPE, BUILDER)                                          \
  BValue convertOp(TYPE op, const TranslationState& state, BuilderBase& fb) { \
    return fb.BUILDER(state.getXlsValue(op.getLhs()),                         \
                      state.getXlsValue(op.getRhs()), state.getLoc(op));      \
  }
XLS_BINARY_OP(AddOp, Add);
XLS_BINARY_OP(SdivOp, SDiv);
XLS_BINARY_OP(SmodOp, SMod);
XLS_BINARY_OP(SubOp, Subtract);
XLS_BINARY_OP(UdivOp, UDiv);
XLS_BINARY_OP(UmodOp, UMod);

#define XLS_MUL_OP(TYPE, BUILDER)                                             \
  BValue convertOp(TYPE op, const TranslationState& state, BuilderBase& fb) { \
    auto result_type =                                                        \
        cast<IntegerType>(getElementTypeOrSelf(op.getResult()));              \
    return fb.BUILDER(state.getXlsValue(op.getLhs()),                         \
                      state.getXlsValue(op.getRhs()), result_type.getWidth(), \
                      state.getLoc(op));                                      \
  }
XLS_MUL_OP(SmulOp, SMul);
XLS_MUL_OP(UmulOp, UMul);

// Partial Products
#define XLS_PARTIAL_PROD_OP(TYPE, BUILDER)                              \
  BValue convertOp(TYPE op, TranslationState& state, BuilderBase& fb) { \
    auto element_type =                                                 \
        cast<IntegerType>(getElementTypeOrSelf(op.getResultLhs()));     \
                                                                        \
    BValue out = fb.BUILDER(state.getXlsValue(op.getLhs()),             \
                            state.getXlsValue(op.getRhs()),             \
                            element_type.getWidth(), state.getLoc(op)); \
    state.setMultiResultValue(op, out, fb);                             \
    return out;                                                         \
  }
XLS_PARTIAL_PROD_OP(SmulpOp, SMulp);
XLS_PARTIAL_PROD_OP(UmulpOp, UMulp);

// Comparison operations
XLS_BINARY_OP(EqOp, Eq);
XLS_BINARY_OP(NeOp, Ne);
XLS_BINARY_OP(SgeOp, SGe);
XLS_BINARY_OP(SgtOp, SGt);
XLS_BINARY_OP(SleOp, SLe);
XLS_BINARY_OP(SltOp, SLt);
XLS_BINARY_OP(UgeOp, UGe);
XLS_BINARY_OP(UgtOp, UGt);
XLS_BINARY_OP(UleOp, ULe);
XLS_BINARY_OP(UltOp, ULt);

// Shift operations
XLS_BINARY_OP(ShllOp, Shll);
XLS_BINARY_OP(ShrlOp, Shrl);
XLS_BINARY_OP(ShraOp, Shra);

// Extension operations
#define XLS_EXTENSION_OP(TYPE, BUILDER)                                       \
  BValue convertOp(TYPE op, const TranslationState& state, BuilderBase& fb) { \
    auto element_type =                                                       \
        cast<IntegerType>(getElementTypeOrSelf(op.getResult()));              \
    return fb.BUILDER(state.getXlsValue(op.getOperand()),                     \
                      element_type.getWidth(), state.getLoc(op));             \
  };
XLS_EXTENSION_OP(ZeroExtOp, ZeroExtend);
XLS_EXTENSION_OP(SignExtOp, SignExtend);

// TODO(jpienaar): Channel operations. This requires more thinking.

// Array operations
BValue convertOp(ArrayOp op, const TranslationState& state, BuilderBase& fb) {
  std::vector<BValue> values;
  for (auto v : op.getInputs()) {
    values.push_back(state.getXlsValue(v));
  }
  assert(!values.empty() && "ArrayOp must have at least one input");
  return fb.Array(values, values.front().GetType(), state.getLoc(op));
}

::xls::Value nestedArrayZero(ArrayType type, BuilderBase& fb) {
  if (auto arrayType = dyn_cast<ArrayType>(type.getElementType())) {
    std::vector<::xls::Value> elements(type.getNumElements(),
                                       nestedArrayZero(arrayType, fb));
    return ::xls::Value::ArrayOwned(std::move(elements));
  }
  if (auto float_type = dyn_cast<FloatType>(type.getElementType())) {
    int mantissa_width = float_type.getFPMantissaWidth() - 1;
    int exponent_width =
        float_type.getWidth() - float_type.getFPMantissaWidth();
    std::vector<::xls::Value> elements = {
        ::xls::Value(::xls::UBits(0, 1)),
        ::xls::Value(::xls::UBits(0, exponent_width)),
        ::xls::Value(::xls::UBits(0, mantissa_width))};
    auto zeroTuple = ::xls::Value::TupleOwned(std::move(elements));
    std::vector<::xls::Value> arrayElements(type.getNumElements(), zeroTuple);
    return ::xls::Value::ArrayOwned(std::move(arrayElements));
  }
  std::vector<uint64_t> zeroes(type.getNumElements());
  return ::xls::Value::UBitsArray(zeroes,
                                  type.getElementType().getIntOrFloatBitWidth())
      .value();
}

BValue convertOp(ArrayZeroOp op, const TranslationState& state,
                 BuilderBase& fb) {
  // TODO(jmolloy): This is only correct for array-of-bits types, not
  // array-of-tuples.
  auto value = nestedArrayZero(op.getType(), fb);
  return fb.Literal(value, state.getLoc(op));
}

BValue convertOp(ArrayIndexOp op, const TranslationState& state,
                 BuilderBase& fb) {
  return fb.ArrayIndex(state.getXlsValue(op.getArray()),
                       {state.getXlsValue(op.getIndex())}, state.getLoc(op));
}

BValue convertOp(ArrayIndexStaticOp op, const TranslationState& state,
                 BuilderBase& fb) {
  constexpr int kIndexBits = 32;  // Just picked arbitrarily.
  return fb.ArrayIndex(state.getXlsValue(op.getArray()),
                       {fb.Literal(::xls::UBits(op.getIndex(), kIndexBits))},
                       state.getLoc(op));
}

BValue convertOp(ArraySliceOp op, const TranslationState& state,
                 BuilderBase& fb) {
  return fb.ArraySlice(state.getXlsValue(op.getArray()),
                       state.getXlsValue(op.getStart()), op.getWidth(),
                       state.getLoc(op));
}

BValue convertOp(ArrayUpdateOp op, const TranslationState& state,
                 BuilderBase& fb) {
  return fb.ArrayUpdate(state.getXlsValue(op.getArray()),
                        state.getXlsValue(op.getValue()),
                        {state.getXlsValue(op.getIndex())}, state.getLoc(op));
}

BValue convertOp(ArrayConcatOp op, const TranslationState& state,
                 BuilderBase& fb) {
  std::vector<BValue> values;
  values.reserve(op.getArrays().size());
  for (auto v : op.getArrays()) {
    values.push_back(state.getXlsValue(v));
  }
  return fb.ArrayConcat(values, state.getLoc(op));
}

BValue convertOp(TraceOp op, const TranslationState& state, BuilderBase& fb) {
  BValue cond;
  if (op.getPredicate()) {
    cond = state.getXlsValue(op.getPredicate());
  } else {
    cond = fb.Literal(::xls::UBits(1, 1));
  }
  std::vector<BValue> args;
  args.reserve(op.getArgs().size());
  for (auto arg : op.getArgs()) {
    args.push_back(state.getXlsValue(arg));
  }
  return fb.Trace(state.getXlsValue(op.getTkn()), cond, args, op.getFormat(),
                  op.getVerbosity(), state.getLoc(op));
}

// Tuple operations
BValue convertOp(TupleOp op, const TranslationState& state, BuilderBase& fb) {
  std::vector<BValue> values;
  for (auto v : op.getOperands()) {
    values.push_back(state.getXlsValue(v));
  }
  return fb.Tuple(values, state.getLoc(op));
}

BValue convertOp(TupleIndexOp op, const TranslationState& state,
                 BuilderBase& fb) {
  return fb.TupleIndex(state.getXlsValue(op.getOperand()), op.getIndex(),
                       state.getLoc(op));
}

// Bit-vector operations
BValue convertOp(BitSliceOp op, const TranslationState& state,
                 BuilderBase& fb) {
  return fb.BitSlice(state.getXlsValue(op.getOperand()),
                     /*start=*/op.getStart(),
                     /*width=*/op.getWidth(), state.getLoc(op));
}

BValue convertOp(BitSliceUpdateOp op, const TranslationState& state,
                 BuilderBase& fb) {
  return fb.BitSliceUpdate(
      state.getXlsValue(op.getOperand()),
      /*start=*/state.getXlsValue(op.getStart()),
      /*update_value=*/state.getXlsValue(op.getUpdateValue()),
      state.getLoc(op));
}

BValue convertOp(DynamicBitSliceOp op, const TranslationState& state,
                 BuilderBase& fb) {
  return fb.DynamicBitSlice(state.getXlsValue(op.getOperand()),
                            /*start=*/state.getXlsValue(op.getStart()),
                            /*width=*/op.getWidth(), state.getLoc(op));
}

BValue convertOp(ConcatOp op, const TranslationState& state, BuilderBase& fb) {
  std::vector<BValue> values;
  for (auto v : op.getOperands()) {
    values.push_back(state.getXlsValue(v));
  }
  return fb.Concat(values, state.getLoc(op));
}

XLS_UNARY_OP(ReverseOp, Reverse);
XLS_UNARY_OP(EncodeOp, Encode);

BValue convertOp(DecodeOp op, const TranslationState& state, BuilderBase& fb) {
  return fb.Decode(state.getXlsValue(op.getOperand()), op.getWidth(),
                   state.getLoc(op));
}

BValue convertOp(OneHotOp op, const TranslationState& state, BuilderBase& fb) {
  return fb.OneHot(
      state.getXlsValue(op.getOperand()), /*priority=*/
      op.getLsbPrio() ? ::xls::LsbOrMsb::kLsb : ::xls::LsbOrMsb::kMsb,
      state.getLoc(op));
};

// Control-oriented operations
BValue convertOp(SelOp op, const TranslationState& state, BuilderBase& fb) {
  std::vector<BValue> cases;
  for (auto v : op.getCases()) {
    cases.push_back(state.getXlsValue(v));
  }
  std::optional<BValue> otherwise;
  if (op.getOtherwise()) {
    otherwise = state.getXlsValue(op.getOtherwise());
  }
  return fb.Select(state.getXlsValue(op.getSelector()), cases, otherwise,
                   state.getLoc(op));
}

BValue convertOp(OneHotSelOp op, const TranslationState& state,
                 BuilderBase& fb) {
  std::vector<BValue> cases;
  for (auto v : op.getCases()) {
    cases.push_back(state.getXlsValue(v));
  }
  return fb.OneHotSelect(state.getXlsValue(op.getSelector()), cases,
                         state.getLoc(op));
}

BValue convertOp(PrioritySelOp op, const TranslationState& state,
                 BuilderBase& fb) {
  std::vector<BValue> cases;
  for (auto v : op.getCases()) {
    cases.push_back(state.getXlsValue(v));
  }
  return fb.PrioritySelect(state.getXlsValue(op.getSelector()), cases,
                           state.getXlsValue(op.getDefaultValue()),
                           state.getLoc(op));
}

FailureOr<PackageInfo> importDslxInstantiation(
    ImportDslxFilePackageOp file_import_op, StringRef dslx_snippet,
    Package& package);

absl::StatusOr<::xls::Function*> GetFunctionFromPackageInfo(
    TranslationState& state, const PackageInfo* package_info,
    const StringRef func_name, const std::string_view package_name,
    const std::string_view func_name_str) {
  if (package_info == nullptr) {
    return state.getPackage().GetFunction(func_name_str);
  }

  // Use the mangled function name to find the XLS function in the package.
  absl::StatusOr<std::string> mangled_func_name =
      ::xls::MangleDslxName(package_name, func_name_str);
  if (!mangled_func_name.ok()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to mangle name: ", mangled_func_name.status().message()));
  }
  // Mangled name is possibly renamed in the merge result, if so, use the new
  // name. This is for the case where one imported multiple packages with the
  // same stem and that have functions with the same names.
  if (auto it =
          package_info->merge_result.name_updates.find(*mangled_func_name);
      it != package_info->merge_result.name_updates.end()) {
    mangled_func_name.emplace(it->second);
  }

  auto fn = state.getPackage().GetFunction(mangled_func_name.value());
  if (fn.ok()) {
    state.addFunction(func_name, fn.value());
  }
  return fn;
}

absl::StatusOr<::xls::Function*> getFunction(TranslationState& state,
                                             const StringRef fn_name) {
  if (auto result = state.getFunction(fn_name)) {
    return result;
  }
  const auto* linkage = state.getLinkage(fn_name);
  bool has_linkage = linkage != nullptr;
  std::string func_name =
      has_linkage ? linkage->getFunction().str() : fn_name.str();

  const PackageInfo* package_info =
      has_linkage
          ? state.getPackageInfo(linkage->getPackage().getLeafReference())
          : nullptr;
  std::string package_name = has_linkage ? package_info->package->name() : "";

  // Handle function instantiations.
  if (func_name.starts_with("fn ")) {
    if (!func_name.ends_with("}")) {
      return absl::InvalidArgumentError(
          "Invalid DSLX snippet, must be a function ending with `}`");
    }
    if (package_info == nullptr) {
      return absl::InvalidArgumentError(
          "Invalid DSLX snippet, must be extern linked");
    }
    auto result = importDslxInstantiation(package_info->op, func_name,
                                          state.getPackage());
    if (failed(result)) {
      return absl::InvalidArgumentError("Failed to import DSLX snippet");
    }
    func_name = StringRef(func_name).drop_front(3).split('(').first.trim();
    return GetFunctionFromPackageInfo(state, &*result, fn_name,
                                      kImportedModuleName, func_name);
  }

  return GetFunctionFromPackageInfo(state, package_info, fn_name, package_name,
                                    func_name);
}

// Converts a float value from a Bits to a tuple of (sign, exponent,
// mantissa).
BValue floatBitsToTuple(BValue float_bits, FloatType float_type,
                        BuilderBase& fb) {
  // Note that getFPMantissaWidth() includes the sign bit.
  int exponent_width = float_type.getWidth() - float_type.getFPMantissaWidth();
  std::vector<BValue> elements = {
      /*Sign*/ fb.BitSlice(float_bits, float_type.getWidth() - 1, 1),
      /*Exponent*/
      fb.BitSlice(float_bits, float_type.getFPMantissaWidth() - 1,
                  exponent_width),
      /*Mantissa*/
      fb.BitSlice(float_bits, 0, float_type.getFPMantissaWidth() - 1),
  };
  return fb.Tuple(elements);
}

BValue coerceFloatResult(Value mlir_result, BValue xls_result,
                         ::xls::Function* func, BuilderBase& fb) {
  if (!isa<FloatType>(mlir_result.getType()) ||
      !xls_result.GetType()->IsBits()) {
    return xls_result;
  }
  assert(func->ForeignFunctionData().has_value());
  // This must be a call to a foreign function that returns a float. The
  // foreign function will return it as a raw bits value, but we expect it
  // as a tuple of (sign, exponent, mantissa).
  auto float_type = cast<FloatType>(mlir_result.getType());
  return floatBitsToTuple(xls_result, float_type, fb);
}

BValue convertOp(func::CallOp call, TranslationState& state, BuilderBase& fb) {
  std::vector<BValue> args;
  for (auto arg : call.getOperands()) {
    args.push_back(state.getXlsValue(arg));
  }
  absl::StatusOr<::xls::Function*> func_or =
      getFunction(state, call.getCallee());
  if (!func_or.ok()) {
    llvm::errs() << "Failed to find function: " << func_or.status().message()
                 << "\n";
    return BValue();
  }
  BValue result = fb.Invoke(args, *func_or, state.getLoc(call));

  if (call.getNumResults() == 1) {
    return coerceFloatResult(call.getResult(0), result, *func_or, fb);
  }

  for (int i = 0, e = call.getNumResults(); i < e; ++i) {
    BValue this_result = fb.TupleIndex(result, i);
    this_result =
        coerceFloatResult(call.getResult(i), this_result, *func_or, fb);
    state.setValue(call.getResult(i), this_result);
  }
  // Just return the tuple result to indicate success.
  return result;
}

BValue convertOp(MapOp mapOp, TranslationState& state, BuilderBase& fb) {
  BValue operand = state.getXlsValue(mapOp.getOperand());
  absl::StatusOr<::xls::Function*> func_or =
      getFunction(state, mapOp.getToApply());
  if (!func_or.ok()) {
    llvm::errs() << "Failed to find function: " << func_or.status().message()
                 << "\n";
    return BValue();
  }
  return fb.Map(operand, *func_or, state.getLoc(mapOp));
}

BValue convertOp(CountedForOp counted_for_op, TranslationState& state,
                 BuilderBase& fb) {
  BValue init = state.getXlsValue(counted_for_op.getInit());
  std::vector<BValue> invar_args;
  for (auto arg : counted_for_op.getInvariantArgs()) {
    invar_args.push_back(state.getXlsValue(arg));
  }
  absl::StatusOr<::xls::Function*> func_or =
      getFunction(state, counted_for_op.getToApply());
  if (!func_or.ok()) {
    llvm::errs() << "Failed to find function: " << func_or.status().message()
                 << "\n";
    return BValue();
  }
  return fb.CountedFor(init, counted_for_op.getTripCount(),
                       counted_for_op.getStride(), *func_or, invar_args,
                       state.getLoc(counted_for_op));
}

::xls::Bits convertAPInt(llvm::APInt apInt) {
  // Doing this in a simple loop, not the most efficient and could be improved
  // if needed (was just avoiding needing to think about the endianness).
  absl::InlinedVector<bool, 64> bits(apInt.getBitWidth());
  for (unsigned i : llvm::seq(0u, apInt.getBitWidth())) {
    bits[i] = apInt[i];
  }
  return ::xls::Bits(bits);
}

// Constant Attribute
::xls::Value convertConstantAttr(Attribute attr) {
  if (auto int_attr = dyn_cast<IntegerAttr>(attr)) {
    auto intType = dyn_cast<IntegerType>(int_attr.getType());
    unsigned bitWidth = intType ? intType.getWidth() : /*IndexType*/ 32u;
    auto intVal = int_attr.getValue();
    absl::InlinedVector<bool, 64> bits(bitWidth);
    // Doing this in a simple loop, not the most efficient and could be improved
    // if needed (was just avoiding needing to think about the endianness).
    for (unsigned i : llvm::seq(0u, bitWidth)) {
      bits[i] = intVal[i];
    }
    return ::xls::Value(::xls::Bits(bits));
  }
  if (auto float_attr = dyn_cast<FloatAttr>(attr)) {
    FloatType float_type = cast<FloatType>(float_attr.getType());
    int mantissa_width = float_type.getFPMantissaWidth() - 1;
    int exponent_width =
        float_type.getWidth() - float_type.getFPMantissaWidth();
    auto apfloat = float_attr.getValue();
    auto apint = apfloat.bitcastToAPInt();
    llvm::APInt sign = apint.getHiBits(1).trunc(1);
    llvm::APInt exponent = apint.extractBits(exponent_width, mantissa_width);
    llvm::APInt mantissa = apint.extractBits(mantissa_width, 0);

    return ::xls::Value::Tuple({
        ::xls::Value(convertAPInt(sign)),
        ::xls::Value(convertAPInt(exponent)),
        ::xls::Value(convertAPInt(mantissa)),
    });
  }
  llvm::errs() << "Unsupported constant type: " << attr << "\n";
  return {};
}

BValue convertOp(ConstantScalarOp op, const TranslationState& state,
                 BuilderBase& fb) {
  if (auto int_attr = dyn_cast<IntegerAttr>(op.getValue())) {
    // TODO(jmolloy): ConstantScalarOp always has I64Attr regardless of the
    // type, so we need special handling here.
    auto int_type = dyn_cast<IntegerType>(op.getType());
    unsigned bit_width = int_type ? int_type.getWidth() : /*IndexType*/ 32u;
    auto int_val = int_attr.getValue().zextOrTrunc(bit_width);
    return fb.Literal(convertAPInt(int_val), state.getLoc(op));
  }
  return fb.Literal(convertConstantAttr(op.getValue()), state.getLoc(op));
}

BValue convertOp(arith::ConstantOp op, const TranslationState& state,
                 BuilderBase& fb) {
  return fb.Literal(convertConstantAttr(op.getValue()), state.getLoc(op));
}

::xls::Value convertLiteralRegion(Block& body, const TranslationState& state,
                                  BuilderBase& fb) {
  std::function<::xls::Value(Operation * op)> convert_op;
  convert_op = [&](Operation* op) {
    return TypeSwitch<Operation*, ::xls::Value>(op)
        .Case<ConstantScalarOp>([&](ConstantScalarOp t) {
          if (auto int_attr = dyn_cast<IntegerAttr>(t.getValue())) {
            // TODO(jmolloy): ConstantScalarOp always has I64Attr regardless of
            // the type, so we need special handling here.
            auto int_type = cast<IntegerType>(t.getType());
            auto int_val = int_attr.getValue().zextOrTrunc(int_type.getWidth());
            return ::xls::Value(convertAPInt(int_val));
          }
          return convertConstantAttr(t.getValue());
        })
        .Case<AfterAllOp>([&](AfterAllOp t) {
          assert(t.getOperands().empty() &&
                 "Only empty after_all permitted in literal");
          return ::xls::Value::Token();
        })
        .Case<TupleOp, ArrayOp>([&](Operation* t) {
          std::vector<::xls::Value> values;
          for (auto v : t->getOperands()) {
            auto value = convert_op(v.getDefiningOp());
            if (value.kind() == ::xls::ValueKind::kInvalid) {
              return ::xls::Value{};
            }
            values.push_back(value);
          }
          if (isa<TupleOp>(op)) {
            return ::xls::Value::Tuple(values);
          } else {
            auto array = ::xls::Value::Array(values);
            if (!array.ok()) {
              return ::xls::Value{};
            } else {
              return array.value();
            }
          }
        })
        .Default([&](auto op) {
          llvm::errs() << "Op not supported inside literal: " << *op << "\n";
          return ::xls::Value();
        });
  };

  return convert_op(body.getTerminator()->getOperand(0).getDefiningOp());
}

BValue convertOp(LiteralOp op, const TranslationState& state, BuilderBase& fb) {
  auto value = convertLiteralRegion(op.getInitializerBlock(), state, fb);
  return fb.Literal(value, state.getLoc(op));
}

// Bitcasts
BValue convertOp(arith::BitcastOp op, const TranslationState& state,
                 BuilderBase& fb) {
  BValue operand = state.getXlsValue(op.getOperand());
  ::xls::Type* expected_type = state.getType(op.getType());
  if (operand.GetType() == expected_type) {
    return operand;
  }

  // Handle conversion from int to float.
  if (isa<FloatType>(op.getType()) &&
      !isa<FloatType>(op.getOperand().getType())) {
    auto float_type = cast<FloatType>(op.getType());
    return floatBitsToTuple(operand, float_type, fb);
  }
  // Handle conversion from float to int.
  if (!isa<FloatType>(op.getType()) &&
      isa<FloatType>(op.getOperand().getType())) {
    return fb.Concat({
        fb.TupleIndex(operand, 0),
        fb.TupleIndex(operand, 1),
        fb.TupleIndex(operand, 2),
    });
  }

  llvm::errs() << "Unsupported bitcast: " << op << "\n";
  return {};
}
BValue convertOp(arith::IndexCastOp op, const TranslationState& state,
                 BuilderBase& fb) {
  // TODO(jmolloy): Check the converted types are the same?
  return state.getXlsValue(op.getOperand());
}

BValue convertOp(AfterAllOp op, const TranslationState& state,
                 BuilderBase& fb) {
  std::vector<BValue> operands;
  for (auto operand : op.getOperands()) {
    operands.push_back(state.getXlsValue(operand));
  }
  return fb.AfterAll(operands);
}

template <typename T>
::xls::Channel* getChannel(T op, const TranslationState& state) {
  auto chan_op = state.getSymbolTable().lookupNearestSymbolFrom<ChanOp>(
      op, op.getChannelAttr());
  if (!chan_op) {
    llvm::errs() << "Channel not found: " << op.getChannelAttr() << "\n";
    return nullptr;
  }
  ::xls::Channel* channel = state.getChannel(chan_op);
  if (!channel) {
    llvm::errs() << "Channel not found: " << chan_op << "\n";
    return nullptr;
  }
  return channel;
}

BValue convertOp(SendOp op, const TranslationState& state, BuilderBase& fb) {
  ProcBuilder* pb = dynamic_cast<ProcBuilder*>(&fb);
  if (!pb) {
    llvm::errs() << "SendOp only supported in procs\n";
    return BValue();
  }
  ::xls::Channel* channel = getChannel(op, state);
  if (!channel) {
    return BValue();
  }
  if (op.getPredicate()) {
    return pb->SendIf(channel, state.getXlsValue(op.getTkn()),
                      state.getXlsValue(op.getPredicate()),
                      state.getXlsValue(op.getData()), state.getLoc(op));
  }
  return pb->Send(channel, state.getXlsValue(op.getTkn()),
                  state.getXlsValue(op.getData()), state.getLoc(op));
}

BValue convertOp(BlockingReceiveOp op, TranslationState& state,
                 BuilderBase& fb) {
  ProcBuilder* pb = dynamic_cast<ProcBuilder*>(&fb);
  if (!pb) {
    llvm::errs() << "BlockingReceiveOp only supported in procs\n";
    return BValue();
  }
  ::xls::Channel* channel = getChannel(op, state);
  if (!channel) {
    return BValue();
  }

  BValue out;
  if (op.getPredicate()) {
    out = pb->ReceiveIf(channel, state.getXlsValue(op.getTkn()),
                        state.getXlsValue(op.getPredicate()), state.getLoc(op));
  } else {
    out =
        pb->Receive(channel, state.getXlsValue(op.getTkn()), state.getLoc(op));
  }
  state.setMultiResultValue(op, out, fb);
  return out;
}

BValue convertOp(NonblockingReceiveOp op, TranslationState& state,
                 BuilderBase& fb) {
  ProcBuilder* pb = dynamic_cast<ProcBuilder*>(&fb);
  if (!pb) {
    llvm::errs() << "NonblockingReceiveOp only supported in procs\n";
    return BValue();
  }
  ::xls::Channel* channel = getChannel(op, state);
  if (!channel) {
    return BValue();
  }
  BValue out;
  if (op.getPredicate()) {
    out = pb->ReceiveIfNonBlocking(channel, state.getXlsValue(op.getTkn()),
                                   state.getXlsValue(op.getPredicate()),
                                   state.getLoc(op));
  } else {
    out = pb->ReceiveNonBlocking(channel, state.getXlsValue(op.getTkn()),
                                 state.getLoc(op));
  }
  state.setMultiResultValue(op, out, fb);
  return out;
}

BValue convertOp(GateOp op, TranslationState& state, BuilderBase& fb) {
  return fb.Gate(state.getXlsValue(op.getCondition()),
                 state.getXlsValue(op.getData()), state.getLoc(op));
}

// Attempts to find the given DSLX file. Tries fileName directory, and also
// tries prepending the runfiles directory.
FailureOr<std::filesystem::path> findDslxFile(
    std::string file_name, std::filesystem::path dslx_search_path) {
  std::vector<std::filesystem::path> candidates = {file_name};
  if (auto run_file_path = ::xls::GetXlsRunfilePath(file_name);
      run_file_path.ok()) {
    candidates.push_back(*run_file_path);
  }

  if (!dslx_search_path.empty()) {
    candidates.push_back(dslx_search_path / file_name);
  }

  for (const auto& candidate : candidates) {
    if (::xls::FileExists(candidate).ok()) {
      return candidate;
    }
  }
  llvm::errs() << "Failed to find DSLX file: " << file_name << "\n";
  return failure();
}

FailureOr<PackageInfo> importDslxInstantiation(
    ImportDslxFilePackageOp file_import_op, StringRef dslx_snippet,
    Package& package) {
  StringRef path = file_import_op.getFilename();

  absl::StatusOr<std::string> package_string_or;

  // Note: this is not bullet proof. The experience if these are wrong would
  // be suboptimal.
  auto fsPath = std::filesystem::path(std::string_view(path));
  if (auto found = findDslxFile(path.str(), fsPath); succeeded(found)) {
    fsPath = *found;
  } else {
    return failure();
  }
  std::string importModule = llvm::join(fsPath.parent_path(), ".");
  std::string packageName = fsPath.stem();
  importModule += "." + packageName;

  // Construct a DSLX module with import.
  std::string dslx;
  llvm::raw_string_ostream os(dslx);
  // Read the file in path into a string.
  std::string file_contents;
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(path);
  if (fileOrErr.getError()) {
    llvm::errs() << "Failed to read file: " << path << "\n";
    return failure();
  }
  os << (*fileOrErr)->getBuffer();
  os << "\n// Imported.\n" << dslx_snippet << "\n";
  os.flush();

  // Note: using a different pathname here else XLS considers this a circular
  // import.
  const ::xls::ConvertDslxToIrOptions options{
      .dslx_stdlib_path = ::xls::GetDefaultDslxStdlibPath(),
      .warnings_as_errors = false,
  };
  package_string_or = ::xls::ConvertDslxToIr(dslx, "<instantiated module>",
                                             kImportedModuleName, options);
  if (!package_string_or.ok()) {
    llvm::errs() << "Failed to convert DSLX to IR: "
                 << package_string_or.status().message() << "\n";
    return failure();
  }
  absl::StatusOr<std::unique_ptr<Package>> package_or =
      ::xls::ParsePackage(package_string_or.value(), std::nullopt);
  if (!package_or.ok()) {
    llvm::errs() << "Failed to parse package: " << package_or.status().message()
                 << "\n";
    return failure();
  }
  absl::StatusOr<Package::PackageMergeResult> merge_result =
      package.ImportFromPackage(package_or->get());
  if (!merge_result.ok()) {
    llvm::errs() << "Failed to add package: " << merge_result.status().message()
                 << "\n";
    return failure();
  }
  return PackageInfo(std::move(package_or.value()), *merge_result,
                     file_import_op);
}

FailureOr<PackageInfo> importDslxFile(ImportDslxFilePackageOp file_import_op,
                                      Package& package,
                                      std::filesystem::path dslx_search_path,
                                      DslxPackageCache& dslx_cache) {
  auto file_name =
      findDslxFile(file_import_op.getFilename().str(), dslx_search_path);
  if (failed(file_name)) {
    return failure();
  }

  absl::StatusOr<std::shared_ptr<const Package>> package_or =
      dslx_cache.import(*file_name, {dslx_search_path});
  if (!package_or.ok()) {
    llvm::errs() << "Failed to parse package: " << package_or.status().message()
                 << "\n";
    return failure();
  }
  absl::StatusOr<Package::PackageMergeResult> merge_result =
      package.ImportFromPackage(package_or->get());
  if (!merge_result.ok()) {
    llvm::errs() << "Failed to add package: " << merge_result.status().message()
                 << "\n";
    return failure();
  }
  return PackageInfo(std::move(package_or.value()), *merge_result,
                     file_import_op);
}

FailureOr<::xls::Value> zeroLiteral(Type type) {
  if (auto int_type = dyn_cast<IntegerType>(type)) {
    return ::xls::Value(convertAPInt(llvm::APInt(int_type.getWidth(), 0)));
  }
  if (auto array_type = dyn_cast<ArrayType>(type)) {
    std::vector<::xls::Value> values;
    values.reserve(array_type.getNumElements());
    for (int i = 0; i < array_type.getNumElements(); ++i) {
      auto value = zeroLiteral(array_type.getElementType());
      if (failed(value)) {
        return failure();
      }
      values.push_back(*value);
    }
    return ::xls::Value::ArrayOwned(std::move(values));
  }
  if (auto tuple_type = dyn_cast<TupleType>(type)) {
    std::vector<::xls::Value> values;
    values.reserve(tuple_type.size());
    for (int i = 0; i < tuple_type.size(); ++i) {
      auto value = zeroLiteral(tuple_type.getType(i));
      if (failed(value)) {
        return failure();
      }
      values.push_back(*value);
    }
    return ::xls::Value::TupleOwned(std::move(values));
  }
  if (auto float_type = dyn_cast<FloatType>(type)) {
    std::vector<::xls::Value> values(3);
    // Note that getFPMantissaWidth() includes the sign bit.
    int exponent_width =
        float_type.getWidth() - float_type.getFPMantissaWidth();
    values[0] = ::xls::Value(convertAPInt(llvm::APInt(1, 0)));
    values[1] = ::xls::Value(convertAPInt(llvm::APInt(exponent_width, 0)));
    values[2] = ::xls::Value(
        convertAPInt(llvm::APInt(float_type.getFPMantissaWidth() - 1, 0)));
    return ::xls::Value::TupleOwned(std::move(values));
  }
  llvm::errs() << "Unsupported type: " << type << "\n";
  return failure();
}

FailureOr<BValue> convertFunction(TranslationState& translation_state,
                                  XlsRegionOpInterface xls_region,
                                  DenseMap<Value, BValue>& value_map,
                                  BuilderBase& fb) {
  // Function return value.
  BValue out;

  // Walk over the function ops and construct the XLS function.
  auto convert = [&](Operation* op) {
    return TypeSwitch<Operation*, BValue>(op)
        .Case<
            // Unary bitwise ops.
            IdentityOp, NotOp,
            // Variadic bitwise operations
            AndOp, NandOp, OrOp, NorOp, XorOp,
            // Bitwise reduction operations
            AndReductionOp, OrReductionOp, XorReductionOp,
            // Arithmetic unary operations
            NegOp,
            // Binary ops.
            AddOp, SdivOp, SmodOp, SmulOp, SmulpOp, SubOp, UdivOp, UmodOp,
            UmulOp, UmulpOp,
            // Comparison operations
            EqOp, NeOp, SgeOp, SgtOp, SleOp, SltOp, UgeOp, UgtOp, UleOp, UltOp,
            // Shift operations
            ShllOp, ShrlOp, ShraOp,
            // Extension operations
            ZeroExtOp, SignExtOp,
            // Array ops.
            ArrayOp, ArrayZeroOp, ArrayIndexOp, ArrayIndexStaticOp,
            ArraySliceOp, ArrayUpdateOp, ArrayConcatOp,
            // Tuple operations
            TupleOp, TupleIndexOp,
            // Bit-vector operations
            BitSliceOp, BitSliceUpdateOp, DynamicBitSliceOp, ConcatOp,
            ReverseOp, DecodeOp, EncodeOp, OneHotOp,
            // Control-oriented operations
            SelOp, OneHotSelOp, PrioritySelOp,
            // Constants
            ConstantScalarOp, arith::ConstantOp, LiteralOp,
            // Control flow
            func::CallOp, CountedForOp, MapOp,
            // Casts
            arith::BitcastOp, arith::IndexCastOp,
            // CSP ops
            AfterAllOp, SendOp, BlockingReceiveOp, NonblockingReceiveOp,
            // Debugging ops
            TraceOp,
            // Misc. side-effecting ops
            GateOp>([&](auto t) { return convertOp(t, translation_state, fb); })
        .Case<func::ReturnOp, YieldOp>([&](auto ret) {
          if (ret.getNumOperands() == 1) {
            return out = value_map[ret.getOperand(0)];
          }
          std::vector<BValue> operands;
          operands.reserve(ret.getNumOperands());
          for (auto operand : ret.getOperands()) {
            operands.push_back(value_map[operand]);
          }
          return out = fb.Tuple(operands);
        })
        .Case<NextValueOp>([&](NextValueOp next) {
          // We just skip the next value op here as its handled in the eproc
          // conversion.
          return value_map[next.getValues()[0]];
        })
        .Case<CallDslxOp>([&](CallDslxOp call) {
          llvm::errs() << "Call remaining, call pass normalize-xls-calls "
                          "before translation\n";
          return BValue();
        })
        .Default([&](auto op) {
          llvm::errs() << "Unsupported op: " << *op << "\n";
          return BValue();
        });
  };

  // Walk over the function and construct the XLS function.
  auto result_walk = xls_region.walk([&](Operation* op) {
    if (op == xls_region) {
      return WalkResult::skip();
    }

    // TODO(schilkp): is there a cleaner way of doing this? Prevents ops inside
    // literal init blocks from getting getting converted individually.
    if (op->getParentOp() && isa<LiteralOp>(op->getParentOp())) {
      return WalkResult::skip();
    }

    // Receives and partial products have multiple results but are explicitly
    // supported.
    if (!isa<BlockingReceiveOp, func::CallOp, NonblockingReceiveOp, UmulpOp,
             SmulpOp>(op)) {
      assert(op->getNumResults() <= 1 && "Multiple results not supported");
    }

    // Handled in the initial region walk.
    if (isa<ChanOp>(op)) {
      return WalkResult::skip();
    }

    if (BValue r = convert(op); r.valid()) {
      if (op->getNumResults() == 1) {
        value_map[op->getResult(0)] = r;
      }
      return WalkResult::advance();
    }
    if (auto s = fb.GetError(); !s.ok()) {
      llvm::errs() << "Construction of conversion failed: " << s.message()
                   << "\n";
    }
    return WalkResult::interrupt();
  });

  if (result_walk.wasInterrupted()) {
    return failure();
  }
  return out;
}

bool isVerilogImport(Operation* op, const ModuleOp& module,
                     const TranslationState& translationState,
                     XlsRegionOpInterface& xlsRegion) {
  bool isImportedVerilog = false;
  if (auto func = dyn_cast<FuncOp>(op)) {
    auto linkage = func->getAttrOfType<TranslationLinkage>("xls.linkage");
    if (linkage) {
      Operation* importOp = translationState.getSymbolTable().lookupSymbolIn(
          module, linkage.getPackage());
      isImportedVerilog = llvm::isa_and_present<ImportVerilogFileOp>(importOp);
    }
  }
  return isImportedVerilog;
}

// This function sets the argument names in the value map, using the
// xls.name attribute.
// Note: In verilog import case the get_name function is not used.
// The ffi verifier insist that there is a parameter in the converted
// xls_func with the name used in the template.
LogicalResult getArgumentNamesForVerilogImport(
    FuncOp& func, XlsRegionOpInterface& xls_region,
    const TranslationState& translation_state, FunctionBuilder& fb,
    ArrayRef<Type> argumentTypes, DenseMap<Value, BValue>& valueMap) {
  // Set the argument names in the value map, using the xls.name
  // attribute.
  // Note: In verilog import case we are using the xls.name attribute to name
  // the XLS function parameter. It is required that this parameter name and the
  // name we use in the FFI template string match.
  ::std::optional<ArrayAttr> argAttrs = func.getArgAttrs();
  const Region::BlockArgListType& argValues =
      xls_region.getBodyRegion().getArguments();
  StringRef attrName = "xls.name";
  ArrayRef<Attribute> attrArgs = argAttrs->getValue();
  for (auto [argValue, attr] : zip(argValues, attrArgs)) {
    auto dictAttr = dyn_cast<DictionaryAttr>(attr);
    auto nameAttr = dictAttr.getNamed(attrName);
    ::xls::Type* xls_type = translation_state.getType(argValue.getType());
    if (!xls_type) {
      return failure();
    }
    valueMap[argValue] =
        fb.Param(cast<StringAttr>(nameAttr->getValue()).str(), xls_type);
  }
  return success();
}

// Generates the code template for the foreign function that is being imported.
// It is used to create the FFI tag for importation of foreign function/verilog.
FailureOr<std::string> generateCodeTemplate(FuncOp func,
                                            ::xls::Function* xlsFunc,
                                            StringRef funcName,
                                            StringRef resultName,
                                            Type resultType) {
  std::string codeTemplate;
  llvm::raw_string_ostream os(codeTemplate);
  os << funcName << " {fn}(";
  auto xlsParams = xlsFunc->params();
  bool added_argument = false;
  llvm::interleaveComma(xlsParams, os, [&](::xls::Param* const param) {
    added_argument = true;
    std::string_view name = param->name();
    os << "." << name << "({" << name << "})";
  });

  if (added_argument) {
    os << ", ";
  }

  if (resultType.isInteger() || isa<FloatType>(resultType)) {
    // Floats tuples are collapsed into a single bits value in verilog.
    os << "." << resultName << "({return}))";
  } else {
    return func->emitError() << "unsupported result type for foreign functions";
  }
  return os.str();
}

LogicalResult annotateForeignFunction(FuncOp func, ::xls::Function& xlsFunc) {
  if (func.getResultTypes().size() != 1) {
    return func->emitError() << "there should be one result when importing"
                                " a verilog function";
  }
  auto linkage = func->getAttrOfType<TranslationLinkage>("xls.linkage");
  if (!linkage || linkage.getKind() != LinkageKind::kForeign) {
    return func->emitError() << "expected foreign xls.linkage attribute";
  }

  StringRef resultName = kResultName;
  auto resultAttrs = func.getResAttrs();
  if (resultAttrs && !resultAttrs->getValue().empty()) {
    auto attributeValues = resultAttrs->getValue();
    for (auto attr : attributeValues) {
      DictionaryAttr dictAttr = dyn_cast<DictionaryAttr>(attr);
      if (!dictAttr) {
        continue;
      }
      std::optional<NamedAttribute> namedAttr = dictAttr.getNamed("xls.name");
      if (namedAttr != std::nullopt) {
        auto nameAttr = cast<StringAttr>(namedAttr->getValue());
        resultName = nameAttr.getValue();
      }
    }
  }

  auto importTemplate =
      generateCodeTemplate(func, &xlsFunc, linkage.getFunction().strref(),
                           resultName, func.getResultTypes().front());
  if (failed(importTemplate)) {
    return failure();
  }

  absl::StatusOr<::xls::ForeignFunctionData> ffd =
      ::xls::ForeignFunctionDataCreateFromTemplate(importTemplate.value());
  if (!ffd.ok()) {
    llvm::errs() << "Failed to create foreign function data: "
                 << ffd.status().message() << "\n";
    return failure();
  }

  xlsFunc.SetForeignFunctionData(ffd.value());
  return success();
}

// If xlsFunc returns a tuple, wraps it into a function that concats the tuple
// elements into a single value and returns that.
//
// XLS natively flattens tuples when generating Verilog, but XLS also wants
// the FFI tag to match the type of the function it's wrapping.
absl::StatusOr<::xls::Function*> wrapDslxFunctionIfNeeded(
    ::xls::Function* xlsFunc, ::xls::Package* package) {
  if (!xlsFunc->GetType()->return_type()->IsTuple()) {
    return xlsFunc;
  }

  xlsFunc = xlsFunc->Clone(xlsFunc->name() + "__bitsreturn__", package).value();

  std::vector<::xls::Node*> nodesToConcat;
  auto* tupleType = xlsFunc->GetType()->return_type()->AsTupleOrDie();
  ::xls::Node* tuple = xlsFunc->return_value();
  nodesToConcat.reserve(tupleType->size());
  for (int i = 0; i < tupleType->size(); ++i) {
    nodesToConcat.push_back(
        xlsFunc->MakeNode<::xls::TupleIndex>(::xls::SourceInfo(), tuple, i)
            .value());
  }
  auto concat =
      xlsFunc->MakeNode<::xls::Concat>(::xls::SourceInfo(), nodesToConcat);
  if (!concat.ok()) {
    return concat.status();
  }
  if (auto status = xlsFunc->set_return_value(*concat); !status.ok()) {
    return status;
  }
  return xlsFunc;
}

::xls::FlopKind convertFlopKind(FlopKind kind) {
  switch (kind) {
    case FlopKind::kNone:
      return ::xls::FlopKind::kNone;
    case FlopKind::kFlop:
      return ::xls::FlopKind::kFlop;
    case FlopKind::kSkid:
      return ::xls::FlopKind::kSkid;
    case FlopKind::kZeroLatency:
      return ::xls::FlopKind::kZeroLatency;
  }
}

FailureOr<std::unique_ptr<Package>> mlirXlsToXls(Operation* op,
                                                 StringRef dslx_search_path,
                                                 DslxPackageCache& dslx_cache) {
  // Treating the outer most module as a package.
  ModuleOp module = dyn_cast<ModuleOp>(op);
  if (!module) {
    return failure();
  }

  // Using either the module name or "_package" as the name of the package.
  // Don't use "package", it's a reserved keyword for the XLS parser.
  std::string module_name =
      module.getName() ? module.getName()->str() : "_package";

  auto package = std::make_unique<Package>(module_name);

  TranslationState translation_state(*package);

  // Translate all channels first as they could be referenced in any order.
  WalkResult walk_result = module.walk([&](ChanOp chan_op) {
    ::xls::Type* xls_type = translation_state.getType(chan_op.getType());
    if (xls_type == nullptr) {
      chan_op.emitOpError("unsupported channel type");
      return WalkResult::interrupt();
    }

    // If this channel is only used by instantiate_eproc ops, we can (and
    // should) skip it. XLS doesn't like orphan channels with no sends or
    // receives.
    auto range = chan_op.getSymbolUses(module);
    if (range.has_value() &&
        llvm::all_of(*range, [](SymbolTable::SymbolUse user) {
          return isa<InstantiateEprocOp>(user.getUser());
        })) {
      return WalkResult::advance();
    }

    std::string name =
        ::xls::verilog::SanitizeVerilogIdentifier(chan_op.getSymName().str());
    ::xls::ChannelOps kind = ::xls::ChannelOps::kSendReceive;
    if (!chan_op.getSendSupported()) {
      kind = ::xls::ChannelOps::kReceiveOnly;
    } else if (!chan_op.getRecvSupported()) {
      kind = ::xls::ChannelOps::kSendOnly;
    }

    std::optional<::xls::FifoConfig> fifo_config = std::nullopt;
    if (auto mlir_fifo_config = chan_op.getFifoConfig()) {
      fifo_config = ::xls::FifoConfig(
          mlir_fifo_config->getFifoDepth(), mlir_fifo_config->getBypass(),
          mlir_fifo_config->getRegisterPushOutputs(),
          mlir_fifo_config->getRegisterPopOutputs());
    }

    std::optional<::xls::FlopKind> input_flop = std::nullopt;
    std::optional<::xls::FlopKind> output_flop = std::nullopt;
    if (auto attr = chan_op.getInputFlopKind()) {
      input_flop = convertFlopKind(*attr);
    }
    if (auto attr = chan_op.getOutputFlopKind()) {
      output_flop = convertFlopKind(*attr);
    }

    auto channel_config =
        ::xls::ChannelConfig(fifo_config, input_flop, output_flop);

    auto channel = package->CreateStreamingChannel(name, kind, xls_type, {},
                                                   channel_config);

    if (!channel.ok()) {
      chan_op.emitOpError("failed to create streaming channel: ")
          << channel.status().message();
      return WalkResult::interrupt();
    }
    translation_state.addChannel(chan_op, *channel);
    return WalkResult::advance();
  });
  if (walk_result.wasInterrupted()) {
    return failure();
  }

  std::filesystem::path search_path(dslx_search_path.str());
  for (auto& op : module.getBodyRegion().front()) {
    // Handle file imports.
    if (auto file_import_op = dyn_cast<ImportDslxFilePackageOp>(op)) {
      auto package_or =
          importDslxFile(file_import_op, *package, search_path, dslx_cache);
      if (failed(package_or)) {
        return failure();
      }
      translation_state.setPackageInfo(file_import_op.getSymName(),
                                       std::move(*package_or));
    }

    // Handle function exports by cloning the linked-to function.
    if (auto export_dslx_op = dyn_cast<ExportDslxOp>(op)) {
      StringRef name = export_dslx_op.getSymName();
      translation_state.addLinkage(name, export_dslx_op.getLinkage());
      // Eagerly import the function.
      auto xlsFunc = getFunction(translation_state, name);
      if (!xlsFunc.ok()) {
        llvm::errs() << "Failed to get function " << name << ": "
                     << xlsFunc.status().message() << "\n";
        return failure();
      }
      if (auto status = (*xlsFunc)->Clone(name, package.get()).status();
          !status.ok()) {
        llvm::errs() << "Failed to clone function " << name << ": "
                     << status.message() << "\n";
        return failure();
      }
      continue;
    }

    // Currently this only works over functions and creates a new XLS function
    // for each function in the module.
    auto xls_region = dyn_cast<XlsRegionOpInterface>(op);
    if (!xls_region) {
      continue;
    }

    // TODO(jpienaar): Do something better here with names.
    DenseMap<Value, std::string> valueNameMap;
    llvm::StringMap<int> usedNames;
    auto get_name = [&](Value v) -> std::string {
      if (auto it = valueNameMap.find(v); it != valueNameMap.end()) {
        return it->second;
      }
      std::string name;
      if (auto loc = dyn_cast<NameLoc>(v.getLoc())) {
        name = ::xls::verilog::SanitizeVerilogIdentifier(loc.getName().str());
      } else {
        name =
            ::xls::verilog::SanitizeVerilogIdentifier(debugString(v.getLoc()));
      }
      auto& count = usedNames[name];
      // If not unique, append counter.
      if (count > 0) {
        name += std::to_string(count);
      }
      ++count;
      return valueNameMap[v] = name;
    };

    bool isImportedVerilog =
        isVerilogImport(&op, module, translation_state, xls_region);
    // Skip function declarations for now.
    if (auto func = dyn_cast<FuncOp>(op); func && func.isDeclaration()) {
      auto linkage = func->getAttrOfType<TranslationLinkage>("xls.linkage");
      if (!linkage) {
        continue;
      }
      translation_state.addLinkage(xls_region.getName(), linkage);
      // Eagerly import the function.
      auto xlsFunc = getFunction(translation_state, xls_region.getName());
      if (!xlsFunc.ok()) {
        llvm::errs() << "Failed to get function " << xls_region.getName()
                     << ": " << xlsFunc.status().message() << "\n";
        return failure();
      }
      if (failed(translation_state.recordOpaqueTypes(func, xlsFunc.value()))) {
        return failure();
      }

      if (linkage.getKind() == LinkageKind::kForeign) {
        xlsFunc = wrapDslxFunctionIfNeeded(xlsFunc.value(), package.get());
        if (!xlsFunc.ok()) {
          llvm::errs() << "Failed to wrap DSLX function "
                       << xls_region.getName() << ": "
                       << xlsFunc.status().message() << "\n";
          return failure();
        }
        translation_state.addFunction(xls_region.getName(), xlsFunc.value());

        auto templateString =
            generateCodeTemplate(func, xlsFunc.value(), xls_region.getName(),
                                 kResultName, func.getResultTypes().front());
        if (failed(templateString)) {
          return failure();
        }
        absl::StatusOr<::xls::ForeignFunctionData> ffd =
            ::xls::ForeignFunctionDataCreateFromTemplate(
                templateString.value());
        if (!ffd.ok()) {
          llvm::errs() << "Failed to create foreign function data: "
                       << ffd.status().message() << "\n";
          return failure();
        }
        xlsFunc.value()->SetForeignFunctionData(ffd.value());
      }
      continue;
    }

    DenseMap<Value, BValue> valueMap;
    translation_state.setValueMap(valueMap);

    if (auto eproc = dyn_cast<EprocOp>(op)) {
      // Populate the state argument values.
      ProcBuilder fb(xls_region.getName(), package.get());
      for (auto arg : xls_region.getBodyRegion().getArguments()) {
        auto literal = zeroLiteral(arg.getType());
        if (failed(literal)) {
          return failure();
        }
        valueMap[arg] = fb.StateElement(get_name(arg), *literal);
      }
      auto out = convertFunction(translation_state, xls_region, valueMap, fb);
      if (failed(out)) {
        return eproc->emitOpError() << "unable to convert eproc";
      }
      for (auto [arg, yield] :
           zip(eproc.getStateArguments(), eproc.getYieldedArguments())) {
        if (auto def =
                dyn_cast_if_present<NextValueOp>(yield.getDefiningOp())) {
          auto next_value = cast<NextValueOp>(def);
          for (auto [pred, value] :
               zip(next_value.getPredicates(), next_value.getValues())) {
            fb.Next(valueMap[arg], valueMap[value], /*pred=*/valueMap[pred]);
          }
        } else {
          fb.Next(valueMap[arg], valueMap[yield]);
        }
      }
      if (absl::StatusOr<::xls::Proc*> s = fb.Build(); !s.ok()) {
        llvm::errs() << "Failed to build proc: " << s.status().message()
                     << "\n";
        return failure();
      }
    } else {
      assert(isa<FuncOp>(op) && "Expected func op");
      FunctionBuilder fb(xls_region.getName(), package.get());
      if (isImportedVerilog) {
        auto func = cast<FuncOp>(op);
        ArrayRef<Type> argumentTypes = func.getArgumentTypes();
        if (failed(getArgumentNamesForVerilogImport(func, xls_region,
                                                    translation_state, fb,
                                                    argumentTypes, valueMap))) {
          return failure();
        }
      } else {
        // Populate the function argument values.
        for (Value arg : xls_region.getBodyRegion().getArguments()) {
          ::xls::Type* xls_type = translation_state.getType(arg.getType());
          if (xls_type == nullptr) {
            return failure();
          }
          valueMap[arg] = fb.Param(get_name(arg), xls_type);
        }
      }
      auto out = convertFunction(translation_state, xls_region, valueMap, fb);
      if (failed(out)) {
        return failure();
      }

      if (absl::StatusOr<::xls::Function*> s = fb.BuildWithReturnValue(*out);
          !s.ok()) {
        llvm::errs() << "Failed to build function: " << s.status().message()
                     << "\n";
        return failure();
      } else {
        // Capture mapping to built function.
        translation_state.addFunction(xls_region.getName(), s.value());
        if (isImportedVerilog) {
          if (failed(annotateForeignFunction(cast<FuncOp>(op), *s.value()))) {
            llvm::errs() << "Failed to annotate foreign function: "
                         << xls_region.getName() << "\n";
            return failure();
          }
        }
      }
    }
  }

  return package;
}

LogicalResult setTop(Operation* op, std::string_view name, Package* package) {
  if (!name.empty()) {
    absl::Status status = package->SetTopByName(name);
    if (!status.ok()) {
      return op->emitError() << "failed to set top: " << status.message();
    }
    return success();
  }

  for (auto region_op :
       cast<ModuleOp>(op).getOps<xls::XlsRegionOpInterface>()) {
    // If name is not set and the function is not private, use the function name
    // as the top.
    if (auto fn = dyn_cast<FuncOp>(*region_op); fn && fn.isPrivate()) {
      continue;
    }
    if (!name.empty()) {
      llvm::errs() << "Multiple potential top functions: " << name << " and "
                   << region_op.getName()
                   << "; pass --main-function to select\n";
      return failure();
    }
    name = region_op.getName();
  }
  if (auto ret = package->SetTopByName(name); !ret.ok()) {
    return op->emitError("failed to set top: ") << ret.message();
  }
  return success();
}
}  // namespace

LogicalResult MlirXlsToXlsTranslate(Operation* op, llvm::raw_ostream& output,
                                    MlirXlsToXlsTranslateOptions options,
                                    MetricsReporter metrics_reporter) {
  // It is important to ensure the XlsDialect is loaded, because it registers
  // the XlsRegionOpInterface external model for FuncOp.
  op->getContext()->getOrLoadDialect<XlsDialect>();
  DslxPackageCache maybe_cache;
  if (options.dslx_cache == nullptr) {
    options.dslx_cache = &maybe_cache;
  }
  if (!options.main_function.empty() && options.privatize_and_dce_functions) {
    op->walk([&](FuncOp func) {
      if (func.isPrivate() || func.getName() == options.main_function) {
        return;
      }
      func.setPrivate();
    });
    PassManager pm(op->getContext());
    pm.addPass(createSymbolDCEPass());
    if (pm.run(op).failed()) {
      return op->emitError("Failed to run SymbolDCE pass");
    }
  }

  auto package =
      mlirXlsToXls(op, options.dslx_search_path, *options.dslx_cache);
  if (failed(package) ||
      failed(setTop(op, options.main_function, package->get()))) {
    return failure();
  }

  if (options.optimize_ir) {
    ::xls::tools::OptOptions opt_options = options.opt_options;
    opt_options.top = (*package)->GetTop().value()->name();
    absl::Status status =
        ::xls::tools::OptimizeIrForTop(package->get(), opt_options);
    if (!status.ok()) {
      llvm::errs() << "Failed to optimize IR: " << status.ToString() << "\n";
      return failure();
    }
  }

  if (!options.generate_verilog) {
    std::string out = (*package)->DumpIr();
    output << out;
    return success();
  }

  auto xls_codegen_results = ::xls::ScheduleAndCodegenPackage(
      package->get(), options.scheduling_options_flags_proto,
      options.codegen_flags_proto,
      /*with_delay_model=*/false);
  if (!xls_codegen_results.ok()) {
    llvm::errs() << "Failed to codegen: "
                 << xls_codegen_results.status().message() << "\n";
    return failure();
  }

  if (metrics_reporter) {
    const ::xls::verilog::XlsMetricsProto& metrics =
        xls_codegen_results->codegen_result.block_metrics;
    if (metrics.has_block_metrics()) {
      const ::xls::verilog::BlockMetricsProto& block_metrics =
          metrics.block_metrics();
      metrics_reporter(**package, block_metrics);
    }
  }

  output << xls_codegen_results->codegen_result.verilog_text;
  return success();
}

absl::StatusOr<std::shared_ptr<const Package>> DslxPackageCache::import(
    const std::string& fileName,
    absl::Span<const std::filesystem::path> additional_search_paths) {
  auto it = cache.find(fileName);
  if (it != cache.end()) {
    return it->second;
  }
  const ::xls::ConvertDslxToIrOptions options{
      .dslx_stdlib_path = ::xls::GetDefaultDslxStdlibPath(),
      .additional_search_paths = additional_search_paths,
      .warnings_as_errors = false,
  };
  absl::StatusOr<std::string> package_string_or =
      ::xls::ConvertDslxPathToIr(fileName, options);
  if (!package_string_or.ok()) {
    return package_string_or.status();
  }
  absl::StatusOr<std::unique_ptr<Package>> package_or =
      ::xls::ParsePackage(package_string_or.value(), std::nullopt);
  if (!package_or.ok()) {
    return package_or.status();
  }

  std::shared_ptr<const Package> package = std::move(package_or.value());
  cache[fileName] = package;
  return package;
}

}  // namespace mlir::xls
