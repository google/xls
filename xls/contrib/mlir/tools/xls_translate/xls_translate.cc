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

#include "xls/contrib/mlir/tools/xls_translate/xls_translate.h"

#include <cassert>
#include <cstdint>
#include <filesystem>  // NOLINT
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
#include "llvm/include/llvm/ADT/STLExtras.h"
#include "llvm/include/llvm/ADT/Sequence.h"
#include "llvm/include/llvm/ADT/StringExtras.h"
#include "llvm/include/llvm/ADT/StringMap.h"
#include "llvm/include/llvm/ADT/StringRef.h"
#include "llvm/include/llvm/ADT/Twine.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"
#include "llvm/include/llvm/Support/Casting.h"  // IWYU pragma: keep
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
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
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/contrib/mlir/IR/xls_ops.h"
#include "xls/ir/bits.h"
#include "xls/ir/source_location.h"
#include "xls/public/function_builder.h"
#include "xls/public/ir.h"
#include "xls/public/ir_parser.h"
#include "xls/public/runtime_build_actions.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/scheduling_options_flags.pb.h"

namespace mlir::xls {
namespace {

using ::mlir::func::FuncOp;
using ::xls::BuilderBase;
using ::xls::BValue;
using ::xls::FunctionBuilder;
using ::xls::Package;
using ::xls::ProcBuilder;

namespace arith = ::mlir::arith;

using ValueMap = DenseMap<Value, BValue>;

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

  void setMultiResultValue(Operation* op, BValue value, BuilderBase& fb) {
    for (auto [i, result] : llvm::enumerate(op->getResults())) {
      BValue extract = fb.TupleIndex(value, i, value.loc());
      value_map_->insert({result, extract});
    }
  }

  const Package& getPackage() const { return package_; }
  Package& getPackage() { return package_; }

  void addFunction(llvm::StringRef name, ::xls::Function* func) {
    func_map_[name] = func;
  }

  void addChannel(ChanOp op, ::xls::Channel* channel) {
    channel_map_[op] = channel;
  }

  ::xls::Channel* getChannel(ChanOp op) const {
    auto it = channel_map_.find(op);
    return it == channel_map_.end() ? nullptr : it->second;
  }

  const PackageInfo* getPackageInfo(llvm::StringRef name) const {
    auto it = package_map_.find(name);
    return it == package_map_.end() ? nullptr : &it->second;
  }

  void setPackageInfo(llvm::StringRef name, PackageInfo&& info) {
    package_map_.try_emplace(name, std::move(info));
  }

  void addLinkage(llvm::StringRef name, TranslationLinkage linkage) {
    linkage_map_[name] = linkage;
  }

  const TranslationLinkage* getLinkage(llvm::StringRef name) const {
    auto it = linkage_map_.find(name);
    return it == linkage_map_.end() ? nullptr : &it->second;
  }

  ::xls::SourceInfo getLoc(Operation* op) const;

  ::xls::Type* getType(Type t);

  LogicalResult recordOpaqueTypes(mlir::func::FuncOp func,
                                  ::xls::Function* xls_func);

  ::xls::Function* getFunction(llvm::StringRef name) const {
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
  llvm::DenseMap<Type, ::xls::Type*> type_map_;

  // Acts as a cache for symbol lookups, should be available even in const
  // functions.
  mutable SymbolTableCollection symbol_table_;

  // For source location
  mutable llvm::StringMap<int> file_numbers_;
  mutable int next_file_number_ = 1;
};

::xls::SourceInfo TranslationState::getLoc(Operation* op) const {
  Location loc = op->getLoc();
  if (isa<mlir::UnknownLoc>(loc)) {
    return ::xls::SourceInfo();
  }
  auto file_loc = dyn_cast<mlir::FileLineColLoc>(loc);
  if (!file_loc) {
    return ::xls::SourceInfo();
  }

  std::filesystem::path path(
      std::string_view(file_loc.getFilename().getValue()));
  int id;
  if (auto it = file_numbers_.find(path.string()); it == file_numbers_.end()) {
    id = next_file_number_++;
    file_numbers_[path.string()] = id;
  } else {
    id = it->second;
  }

  return ::xls::SourceInfo(::xls::SourceLocation(
      ::xls::Fileno(id), ::xls::Lineno(file_loc.getLine()),
      ::xls::Colno(file_loc.getColumn())));
}

::xls::Type* TranslationState::getType(Type t) {
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

LogicalResult TranslationState::recordOpaqueTypes(mlir::func::FuncOp func,
                                                  ::xls::Function* xls_func) {
  ::xls::FunctionType* xls_func_type = xls_func->GetType();  // NOLINT
  for (auto [mlirType, xls_type] :
       llvm::zip(func.getArgumentTypes(), xls_func_type->parameters())) {
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
  for (auto [mlirType, xls_type] :
       llvm::zip(func.getResultTypes(), return_types)) {
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
XLS_VARIADIC_BINARY_OP(OrOp, Or);
XLS_VARIADIC_BINARY_OP(XorOp, Xor);

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
XLS_BINARY_OP(SmulOp, SMul);
XLS_BINARY_OP(SubOp, Subtract);
XLS_BINARY_OP(UdivOp, UDiv);
XLS_BINARY_OP(UmodOp, UMod);
XLS_BINARY_OP(UmulOp, UMul);

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
        cast<IntegerType>(mlir::getElementTypeOrSelf(op.getResult()));        \
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

::xls::Value nestedArrayZero(ArrayType type, BuilderBase& fb) {  // NOLINT
  if (auto arrayType = dyn_cast<ArrayType>(type.getElementType())) {
    std::vector<::xls::Value> elements(type.getNumElements(),  // NOLINT
                                       nestedArrayZero(arrayType, fb));
    return ::xls::Value::ArrayOwned(std::move(elements));  // NOLINT
  }
  if (auto float_type = dyn_cast<FloatType>(type.getElementType())) {
    int mantissa_width = float_type.getFPMantissaWidth() - 1;
    int exponent_width =
        float_type.getWidth() - float_type.getFPMantissaWidth();
    // NOLINTNEXTLINE
    std::vector<::xls::Value> elements = {
        ::xls::Value(::xls::UBits(0, 1)),                            // NOLINT
        ::xls::Value(::xls::UBits(0, exponent_width)),               // NOLINT
        ::xls::Value(::xls::UBits(0, mantissa_width))};              // NOLINT
    auto zeroTuple = ::xls::Value::TupleOwned(std::move(elements));  // NOLINT
    std::vector<::xls::Value> arrayElements(type.getNumElements(),   // NOLINT
                                            zeroTuple);
    return ::xls::Value::ArrayOwned(std::move(arrayElements));  // NOLINT
  }
  std::vector<uint64_t> zeroes(type.getNumElements());
  return ::xls::Value::UBitsArray(  // NOLINT
             zeroes, type.getElementType().getIntOrFloatBitWidth())
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
                       // NOLINTNEXTLINE
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
      // NOLINTNEXTLINE
      op.getLsbPrio() ? ::xls::LsbOrMsb::kLsb : ::xls::LsbOrMsb::kMsb,
      state.getLoc(op));
};

// Control-oriented operations
BValue convertOp(SelOp op, const TranslationState& state, BuilderBase& fb) {
  std::vector<BValue> cases;
  for (auto v : op.getCases()) {
    cases.push_back(state.getXlsValue(v));
  }
  return fb.Select(state.getXlsValue(op.getSelector()), cases,
                   state.getXlsValue(op.getOtherwise()), state.getLoc(op));
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
    ImportDslxFilePackageOp file_import_op, llvm::StringRef dslx_snippet,
    Package& package);

absl::StatusOr<::xls::Function*> getFunction(TranslationState& state,
                                             const llvm::StringRef fn_name) {
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
    if (failed(importDslxInstantiation(package_info->op, func_name,
                                       state.getPackage()))) {
      return absl::InvalidArgumentError("Failed to import DSLX snippet");
    }
    func_name =
        llvm::StringRef(func_name).drop_front(3).split('(').first.trim();
  }

  if (package_info == nullptr) {
    return state.getPackage().GetFunction(func_name);
  }

  // Use the mangled function name to find the XLS function in the package.
  absl::StatusOr<std::string> mangled_func_name =
      ::xls::MangleDslxName(package_info->package->name(), func_name);
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
    state.addFunction(fn_name, fn.value());
  }
  return fn;
}

BValue convertOp(mlir::func::CallOp call, TranslationState& state,
                 BuilderBase& fb) {
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
  return fb.Invoke(args, *func_or, state.getLoc(call));
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

// NOLINTNEXTLINE
::xls::Bits convertAPInt(llvm::APInt apInt) {
  // Doing this in a simple loop, not the most efficient and could be improved
  // if needed (was just avoiding needing to think about the endianness).
  absl::InlinedVector<bool, 64> bits(apInt.getBitWidth());
  for (unsigned i : llvm::seq(0u, apInt.getBitWidth())) {
    bits[i] = apInt[i];
  }
  // NOLINTNEXTLINE
  return ::xls::Bits(bits);
}

// Constant operation
BValue convertConstantAttr(Attribute attr, const TranslationState& /*state*/,
                           BuilderBase& fb) {
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
    // NOLINTNEXTLINE
    return fb.Literal(::xls::Bits(bits));
  }
  if (auto float_attr = dyn_cast<FloatAttr>(attr)) {
    mlir::FloatType float_type = cast<mlir::FloatType>(float_attr.getType());
    int mantissa_width = float_type.getFPMantissaWidth() - 1;
    int exponent_width =
        float_type.getWidth() - float_type.getFPMantissaWidth();
    auto apfloat = float_attr.getValue();
    auto apint = apfloat.bitcastToAPInt();
    llvm::APInt sign = apint.getHiBits(1).trunc(1);
    llvm::APInt mantissa = apint.extractBits(mantissa_width, exponent_width);
    llvm::APInt exponent = apint.extractBits(exponent_width, 0);
    return fb.Tuple({fb.Literal(convertAPInt(sign)),
                     fb.Literal(convertAPInt(exponent)),
                     fb.Literal(convertAPInt(mantissa))});
  }
  llvm::errs() << "Unsupported constant type: " << attr << "\n";
  return {};
}

BValue convertOp(ConstantScalarOp op, const TranslationState& state,
                 BuilderBase& fb) {
  if (auto int_attr = dyn_cast<IntegerAttr>(op.getValue())) {
    // TODO(jmolloy): ConstantScalarOp always has I64Attr regardless of the
    // type, so we need special handling here.
    auto int_type = cast<IntegerType>(op.getType());
    auto int_val = int_attr.getValue().zextOrTrunc(int_type.getWidth());
    return fb.Literal(convertAPInt(int_val), state.getLoc(op));
  }
  return convertConstantAttr(op.getValue(), state, fb);
}
BValue convertOp(arith::ConstantOp op, const TranslationState& state,
                 BuilderBase& fb) {
  return convertConstantAttr(op.getValue(), state, fb);
}

// Bitcasts
BValue convertOp(arith::BitcastOp op, const TranslationState& state,
                 BuilderBase& fb) {
  // TODO(jmolloy): Check the converted types are the same?
  return state.getXlsValue(op.getOperand());
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
    llvm::errs() << "SendOp only supported in procs\n";
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
    llvm::errs() << "SendOp only supported in procs\n";
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

FailureOr<PackageInfo> importDslxInstantiation(
    ImportDslxFilePackageOp file_import_op, llvm::StringRef dslx_snippet,
    Package& package) {
  ::llvm::StringRef path = file_import_op.getFilename();

  std::string module_name = "imported_module";
  std::string_view stdlib_path = ::xls::GetDefaultDslxStdlibPath();
  std::vector<std::filesystem::path> additional_search_paths = {};
  absl::StatusOr<std::string> package_string_or;

  // Note: this is not bullet proof. The experience if these are wrong would
  // be suboptimal.
  std::string importModule = llvm::join(
      std::filesystem::path(std::string_view(path)).parent_path(), ".");
  importModule +=
      "." + std::filesystem::path(std::string_view(path)).stem().string();

  // Construct a DSLX module with import.
  std::string dslx;
  llvm::raw_string_ostream os(dslx);
  os << "import " << importModule << " as im;\n";
  for (llvm::StringRef x : llvm::split(dslx_snippet, 0x7B)) {
    os << x << (x.ends_with("}") ? "" : "{ im::");
  }
  os.flush();

  // Note: using a different pathname here else XLS considers this a circular
  // import.
  package_string_or =
      ::xls::ConvertDslxToIr(dslx, "<instantiated module>", module_name,
                             stdlib_path, additional_search_paths);
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
      package.AddPackage(package_or->get());
  if (!merge_result.ok()) {
    llvm::errs() << "Failed to add package: " << merge_result.status().message()
                 << "\n";
    return failure();
  }
  return PackageInfo(std::move(package_or.value()), *merge_result,
                     file_import_op);
}

// Attempts to find the given DSLX file. Tries fileName directory, and also
// tries prepending the runfiles directory.
FailureOr<std::filesystem::path> findDslxFile(
    std::string file_name, llvm::StringRef dslx_search_path) {
  std::vector<std::filesystem::path> candidates = {file_name};
  if (auto run_file_path = ::xls::GetXlsRunfilePath(file_name);
      run_file_path.ok()) {
    candidates.push_back(*run_file_path);
  }

  if (!dslx_search_path.empty()) {
    candidates.push_back(std::filesystem::path(dslx_search_path.str()) /
                         file_name);
  }

  for (const auto& candidate : candidates) {
    if (::xls::FileExists(candidate).ok()) {
      return candidate;
    }
  }
  llvm::errs() << "Failed to find DSLX file: " << file_name << "\n";
  return failure();
}

FailureOr<PackageInfo> importDslxFile(ImportDslxFilePackageOp file_import_op,
                                      Package& package,
                                      llvm::StringRef dslx_search_path,
                                      DslxPackageCache& dslx_cache) {
  auto file_name =
      findDslxFile(file_import_op.getFilename().str(), dslx_search_path);
  if (failed(file_name)) {
    return failure();
  }

  absl::StatusOr<std::shared_ptr<const Package>> package_or =
      dslx_cache.import(*file_name);
  if (!package_or.ok()) {
    llvm::errs() << "Failed to parse package: " << package_or.status().message()
                 << "\n";
    return failure();
  }
  absl::StatusOr<Package::PackageMergeResult> merge_result =
      package.AddPackage(package_or->get());
  if (!merge_result.ok()) {
    llvm::errs() << "Failed to add package: " << merge_result.status().message()
                 << "\n";
    return failure();
  }
  return PackageInfo(std::move(package_or.value()), *merge_result,
                     file_import_op);
}

// NOLINTNEXTLINE
FailureOr<::xls::Value> zeroLiteral(Type type) {
  if (auto int_type = dyn_cast<IntegerType>(type)) {
    // NOLINTNEXTLINE
    return ::xls::Value(convertAPInt(llvm::APInt(int_type.getWidth(), 0)));
  }
  if (auto array_type = dyn_cast<ArrayType>(type)) {
    std::vector<::xls::Value> values;  // NOLINT
    values.reserve(array_type.getNumElements());
    for (int i = 0; i < array_type.getNumElements(); ++i) {
      auto value = zeroLiteral(array_type.getElementType());
      if (failed(value)) {
        return failure();
      }
      values.push_back(*value);
    }
    return ::xls::Value::ArrayOwned(std::move(values));  // NOLINT
  }
  if (auto tuple_type = dyn_cast<TupleType>(type)) {
    std::vector<::xls::Value> values;  // NOLINT
    values.reserve(tuple_type.size());
    for (int i = 0; i < tuple_type.size(); ++i) {
      auto value = zeroLiteral(tuple_type.getType(i));
      if (failed(value)) {
        return failure();
      }
      values.push_back(*value);
    }
    return ::xls::Value::TupleOwned(std::move(values));  // NOLINT
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

  // Walk over the function and construct the XLS function.
  auto convert = [&](Operation* op) {
    return TypeSwitch<Operation*, BValue>(op)
        .Case<
            // Unary bitwise ops.
            IdentityOp, NotOp,
            // Variadic bitwise operations
            AndOp, OrOp, XorOp,
            // Arithmetic unary operations
            NegOp,
            // Binary ops.
            AddOp, SdivOp, SmodOp, SmulOp, SubOp, UdivOp, UmodOp, UmulOp,
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
            ConstantScalarOp, arith::ConstantOp,
            // Control flow
            mlir::func::CallOp, CountedForOp, MapOp,
            // Casts
            arith::BitcastOp, arith::IndexCastOp,
            // CSP ops
            AfterAllOp, SendOp, BlockingReceiveOp, NonblockingReceiveOp,
            // Debugging ops
            TraceOp>(
            [&](auto t) { return convertOp(t, translation_state, fb); })
        .Case<mlir::func::ReturnOp, YieldOp>([&](auto ret) {
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
    // Receives have multiple results but are explicitly supported.
    if (!isa<BlockingReceiveOp, NonblockingReceiveOp>(op)) {
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

FailureOr<std::unique_ptr<Package>> mlirXlsToXls(
    Operation* op, llvm::StringRef dslx_search_path,
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
    std::string name = chan_op.getSymName().str();
    ::xls::ChannelOps kind = ::xls::ChannelOps::kSendReceive;  // NOLINT
    if (!chan_op.getSendSupported()) {
      kind = ::xls::ChannelOps::kReceiveOnly;  // NOLINT
    } else if (!chan_op.getRecvSupported()) {
      kind = ::xls::ChannelOps::kSendOnly;  // NOLINT
    }
    auto channel = package->CreateStreamingChannel(name, kind, xls_type);
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

  for (auto& op : module.getBodyRegion().front()) {
    // Handle file imports.
    if (auto file_import_op = dyn_cast<ImportDslxFilePackageOp>(op)) {
      auto package_or = importDslxFile(file_import_op, *package,
                                       dslx_search_path, dslx_cache);
      if (failed(package_or)) {
        return failure();
      }
      translation_state.setPackageInfo(file_import_op.getSymName(),
                                       std::move(*package_or));
    }

    // Currently this only works over functions and creates a new XLS function
    // for each function in the module.
    if (auto xls_region = dyn_cast<XlsRegionOpInterface>(op)) {
      // Skip function declarations for now.
      if (auto func = dyn_cast<FuncOp>(op); func && func.isDeclaration()) {
        if (auto linkage =
                func->getAttrOfType<TranslationLinkage>("xls.linkage")) {
          translation_state.addLinkage(xls_region.getName(), linkage);
          // Eagerly import the function.
          auto xlsFunc = getFunction(translation_state, xls_region.getName());
          if (!xlsFunc.ok()) {
            llvm::errs() << "Failed to get function " << xls_region.getName()
                         << ": " << xlsFunc.status().message() << "\n";
            return failure();
          }
          if (failed(
                  translation_state.recordOpaqueTypes(func, xlsFunc.value()))) {
            return failure();
          }
        }
        continue;
      }

      // TODO(jpienaar): Do something better here with names.
      auto get_name = [&](Value v) -> std::string {
        return mlir::debugString(v.getLoc());
      };
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
        std::vector<BValue> next_state;
        for (Value arg : eproc.getYieldedArguments()) {
          next_state.push_back(valueMap[arg]);
        }
        if (absl::StatusOr<::xls::Proc*> s = fb.Build(next_state); !s.ok()) {
          llvm::errs() << "Failed to build proc: " << s.status().message()
                       << "\n";
          return failure();
        }
      } else {
        assert(isa<FuncOp>(op) && "Expected func op");
        FunctionBuilder fb(xls_region.getName(), package.get());

        // Populate the function argument values.
        for (Value arg : xls_region.getBodyRegion().getArguments()) {
          ::xls::Type* xls_type = translation_state.getType(arg.getType());
          if (xls_type == nullptr) {
            return failure();
          }
          valueMap[arg] = fb.Param(get_name(arg), xls_type);
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
                                    MlirXlsToXlsTranslateOptions options) {
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
    mlir::PassManager pm(op->getContext());
    pm.addPass(mlir::createSymbolDCEPass());
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

  std::string out = (*package)->DumpIr();
  if (options.optimize_ir) {
    auto optimized =
        ::xls::OptimizeIr(out, (*package)->GetTop().value()->name());
    if (!optimized.ok()) {
      llvm::errs() << "Failed to optimize IR: " << optimized.status().message()
                   << "\n";
      return failure();
    }
    out = *optimized;
  }

  if (!options.generate_verilog) {
    output << out;
    return success();
  }

  ::xls::SchedulingOptionsFlagsProto scheduling_options_flags_proto;
  ::xls::CodegenFlagsProto codegen_flags_proto;
  codegen_flags_proto.set_generator(::xls::GENERATOR_KIND_COMBINATIONAL);
  codegen_flags_proto.set_register_merge_strategy(
      ::xls::RegisterMergeStrategyProto::STRATEGY_DONT_MERGE);

  auto xls_codegen_results = ::xls::ScheduleAndCodegenPackage(
      &**package, scheduling_options_flags_proto, codegen_flags_proto,
      /*with_delay_model=*/false);
  if (!xls_codegen_results.ok()) {
    llvm::errs() << "Failed to codegen: "
                 << xls_codegen_results.status().message() << "\n";
    return failure();
  }

  output << xls_codegen_results->module_generator_result.verilog_text;
  return success();
}

absl::StatusOr<std::shared_ptr<const Package>> DslxPackageCache::import(
    const std::string& fileName) {
  auto it = cache.find(fileName);
  if (it != cache.end()) {
    return it->second;
  }
  absl::StatusOr<std::string> package_string_or = ::xls::ConvertDslxPathToIr(
      fileName, ::xls::GetDefaultDslxStdlibPath(), {});
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
