// Copyright 2020 Google LLC
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

#include "xls/ir/llvm_ir_jit.h"

#include <cstddef>
#include <memory>

#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm-c/Target.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "xls/codegen/vast.h"
#include "xls/common/integral_types.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/logging/vlog_is_on.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/keyword_args.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_helpers.h"
namespace xls {
namespace {

// Convenience alias for XLS type => LLVM type mapping used as a cache.
using TypeCache = absl::flat_hash_map<const Type*, llvm::Type*>;

// Visitor to construct LLVM IR for each encountered XLS IR node. Based on
// DfsVisitorWithDefault to highlight any unhandled IR nodes.
class BuilderVisitor : public DfsVisitorWithDefault {
 public:
  // llvm_entry_function is the function being used to enter "LLVM space", not
  // the entry function to the XLS Package. It's necessary to know for parameter
  // handling (whether or not we handle params as normal LLVM values or values
  // read from an input char buffer).
  explicit BuilderVisitor(llvm::Module* module, llvm::IRBuilder<>* builder,
                          absl::Span<Param* const> params,
                          absl::optional<Function*> llvm_entry_function,
                          LlvmTypeConverter* type_converter)
      : module_(module),
        context_(&module_->getContext()),
        builder_(builder),
        return_value_(nullptr),
        type_converter_(type_converter),
        llvm_entry_function_(llvm_entry_function) {
    for (int i = 0; i < params.size(); ++i) {
      int64 start = i == 0 ? 0 : arg_indices_[i - 1].second + 1;
      int64 end =
          start + type_converter->GetTypeByteSize(*params[i]->GetType()) - 1;
      arg_indices_.push_back({start, end});
    }
  }

  absl::Status DefaultHandler(Node* node) override {
    return absl::UnimplementedError(
        absl::StrCat("Unhandled node: ", node->ToString()));
  }

  absl::Status HandleAdd(BinOp* binop) override { return HandleBinOp(binop); }

  absl::Status HandleAndReduce(BitwiseReductionOp* op) override {
    // AND-reduce is equivalent to checking if every bit is set in the input.
    llvm::Value* operand = node_map_.at(op->operand(0));
    llvm::IntegerType* operand_type =
        llvm::cast<llvm::IntegerType>(operand->getType());
    llvm::Value* eq = builder_->CreateICmpEQ(
        operand, llvm::ConstantInt::get(operand_type, operand_type->getMask()));
    return StoreResult(op, eq);
  }

  absl::Status HandleAfterAll(AfterAll* after_all) override {
    // AfterAll is only meaningful to the compiler and does not actually perform
    // any computation. Furter, token types don't contain any data. A 0-element
    // array is convenient and low-overhead way to let the rest of the llvm
    // infrastructure treat token like a normal data-type.
    return StoreResult(
        after_all,
        llvm::ConstantArray::get(
            llvm::ArrayType::get(llvm::IntegerType::get(*context_, 1), 0),
            llvm::ArrayRef<llvm::Constant*>()));
  }

  absl::Status HandleArray(Array* array) override {
    llvm::Type* array_type =
        type_converter_->ConvertToLlvmType(*array->GetType());

    llvm::Value* result = CreateTypedZeroValue(array_type);
    for (uint32 i = 0; i < array->size(); ++i) {
      result = builder_->CreateInsertValue(
          result, node_map_.at(array->operand(i)), {i});
    }

    return StoreResult(array, result);
  }

  absl::Status HandleArrayIndex(ArrayIndex* index) override {
    // Get the pointer to the element of interest, then load it. Easy peasy.
    llvm::Value* array = node_map_.at(index->operand(0));
    llvm::Value* index_value = node_map_.at(index->operand(1));
    int64 index_width = index_value->getType()->getIntegerBitWidth();

    // Our IR does not use negative indices, so we add a
    // zero MSb to prevent LLVM from interpreting this as such.
    std::vector<llvm::Value*> gep_indices = {
        llvm::ConstantInt::get(llvm::Type::getInt64Ty(*context_), 0),
        builder_->CreateZExt(
            index_value, llvm::IntegerType::get(*context_, index_width + 1))};

    // Ideally, we'd use IRBuilder::CreateExtractValue here, but that requires
    // constant indices. Since there's no other way to extract a value from an
    // aggregate, we're left with storing the value in a temporary alloca and
    // using that pointer to extract the value.
    llvm::AllocaInst* alloca;
    if (!array_storage_.contains(array)) {
      alloca = builder_->CreateAlloca(array->getType());
      builder_->CreateStore(array, alloca);
      array_storage_[array] = alloca;
    } else {
      alloca = array_storage_[array];
    }

    llvm::Value* gep = builder_->CreateGEP(alloca, gep_indices);
    return StoreResult(index, builder_->CreateLoad(gep));
  }

  absl::Status HandleArrayUpdate(ArrayUpdate* update) override {
    llvm::Value* original_array = node_map_.at(update->operand(0));
    llvm::Type* array_type = original_array->getType();
    llvm::Value* index_value = node_map_.at(update->operand(1));
    llvm::Value* update_value = node_map_.at(update->operand(2));
    llvm::AllocaInst* alloca = builder_->CreateAlloca(array_type);
    builder_->CreateStore(original_array, alloca);

    // We must compare the index to the size of the array. Both arguments
    // for this comparison must have the same bitwidth, so we will cast the
    // arguments using the maximum bitwidth of the two. Value::size()  - used
    // to get the array size  - returns an int64, so the bitwidth of the array
    // size can be no larger than 64 bits. The index could have an arbitrarily
    // large bitwidth.
    int64 index_bitwidth = index_value->getType()->getIntegerBitWidth();
    int64 comparison_bitwidth = std::max(index_bitwidth, (int64)64);
    llvm::Value* array_size_comparison_bitwidth = llvm::ConstantInt::get(
        llvm::Type::getIntNTy(*context_, comparison_bitwidth), update->size());
    llvm::Value* index_value_comparison_bitwidth = builder_->CreateZExt(
        index_value, llvm::Type::getIntNTy(*context_, comparison_bitwidth));
    llvm::Value* index_inbounds = builder_->CreateICmpULT(
        index_value_comparison_bitwidth, array_size_comparison_bitwidth);

    // Update array.
    llvm::Value* bounds_safe_index_value = builder_->CreateSelect(
        index_inbounds, index_value,
        llvm::ConstantInt::get(index_value->getType(), 0));
    // Our IR does not use negative indices, so we add a
    // zero MSb to prevent LLVM from interpreting this as such.
    std::vector<llvm::Value*> gep_indices = {
        llvm::ConstantInt::get(llvm::Type::getInt64Ty(*context_), 0),
        builder_->CreateZExt(
            bounds_safe_index_value,
            llvm::IntegerType::get(*context_, index_bitwidth + 1))};
    llvm::Value* gep = builder_->CreateGEP(alloca, gep_indices);
    llvm::Value* original_element_value = builder_->CreateLoad(gep);
    llvm::Value* bounds_safe_update_value = builder_->CreateSelect(
        index_inbounds, update_value, original_element_value);
    builder_->CreateStore(bounds_safe_update_value, gep);

    llvm::Value* update_array = builder_->CreateLoad(array_type, alloca);
    // Record allocated memory for updated array.
    if (array_storage_.contains(update_array)) {
      return absl::InternalError(absl::StrFormat(
          "Newly created update array %s was already allocated memory somehow.",
          update->ToString()));
    }
    array_storage_[update_array] = alloca;

    return StoreResult(update, update_array);
  }

  absl::Status HandleBitSlice(BitSlice* bit_slice) override {
    llvm::Value* value = node_map_.at(bit_slice->operand(0));
    Value shift_amount(
        UBits(bit_slice->start(), value->getType()->getIntegerBitWidth()));
    XLS_ASSIGN_OR_RETURN(
        llvm::Constant * start,
        type_converter_->ToLlvmConstant(value->getType(), shift_amount));

    // Then shift and "mask" (by casting) the input value.
    llvm::Value* shifted_value = builder_->CreateLShr(value, start);
    llvm::Value* truncated_value = builder_->CreateTrunc(
        shifted_value, llvm::IntegerType::get(*context_, bit_slice->width()));
    return StoreResult(bit_slice, truncated_value);
  }

  absl::Status HandleDynamicBitSlice(
      DynamicBitSlice* dynamic_bit_slice) override {
    llvm::Value* value = node_map_.at(dynamic_bit_slice->operand(0));
    llvm::Value* start = node_map_.at(dynamic_bit_slice->operand(1));
    int64 value_width = value->getType()->getIntegerBitWidth();
    int64 start_width = start->getType()->getIntegerBitWidth();
    // Either value or start may be wider, so we use the widest of both
    // since LLVM requires both arguments to be of the same type for
    // comparison and shifting.
    int64 max_width = std::max(start_width, value_width);
    llvm::IntegerType* max_width_type = builder_->getIntNTy(max_width);
    llvm::Value* value_ext = builder_->CreateZExt(value, max_width_type);
    llvm::Value* start_ext = builder_->CreateZExt(start, max_width_type);

    Value operand_width(UBits(value_width, max_width));
    XLS_ASSIGN_OR_RETURN(
        llvm::Constant * bit_width,
        type_converter_->ToLlvmConstant(max_width_type, operand_width));

    // "out_of_bounds" indicates whether slice is completely out of bounds
    llvm::Value* out_of_bounds = builder_->CreateICmpUGE(start_ext, bit_width);
    llvm::IntegerType* return_type =
        llvm::IntegerType::get(*context_, dynamic_bit_slice->width());
    XLS_ASSIGN_OR_RETURN(
        llvm::Constant * zeros,
        type_converter_->ToLlvmConstant(
            return_type, Value(Bits(dynamic_bit_slice->width()))));
    // Then shift and truncate the input value.
    llvm::Value* shifted_value = builder_->CreateLShr(value_ext, start_ext);
    llvm::Value* truncated_value =
        builder_->CreateTrunc(shifted_value, return_type);
    llvm::Value* result =
        builder_->CreateSelect(out_of_bounds, zeros, truncated_value);
    return StoreResult(dynamic_bit_slice, result);
  }

  absl::Status HandleConcat(Concat* concat) override {
    llvm::Type* dest_type =
        type_converter_->ConvertToLlvmType(*concat->GetType());
    llvm::Value* base = llvm::ConstantInt::get(dest_type, 0);

    int current_shift = dest_type->getIntegerBitWidth();
    for (const Node* xls_operand : concat->operands()) {
      // Widen each operand to the full size, shift to the right location, and
      // bitwise or into the result value.
      int64 operand_width = xls_operand->BitCountOrDie();
      llvm::Value* operand = node_map_.at(xls_operand);
      operand = builder_->CreateZExt(operand, dest_type);
      llvm::Value* shifted_operand =
          builder_->CreateShl(operand, current_shift - operand_width);
      base = builder_->CreateOr(base, shifted_operand);

      current_shift -= operand_width;
    }

    return StoreResult(concat, base);
  }

  absl::Status HandleCountedFor(CountedFor* counted_for) override {
    XLS_ASSIGN_OR_RETURN(llvm::Function * function,
                         GetModuleFunction(counted_for->body()));
    // One for the loop carry and one for the index.
    std::vector<llvm::Value*> args(counted_for->invariant_args().size() + 2);
    for (int i = 0; i < counted_for->invariant_args().size(); i++) {
      args[i + 2] = node_map_.at(counted_for->invariant_args()[i]);
    }
    args[1] = node_map_.at(counted_for->initial_value());

    llvm::Type* function_type = function->getType()->getPointerElementType();
    for (int i = 0; i < counted_for->trip_count(); ++i) {
      args[0] = llvm::ConstantInt::get(function_type->getFunctionParamType(0),
                                       i * counted_for->stride());
      args[1] = builder_->CreateCall(function, {args});
    }

    return StoreResult(counted_for, args[1]);
  }

  absl::Status HandleDecode(Decode* decode) override {
    llvm::Value* input = node_map_.at(decode->operand(0));
    llvm::Type* result_type =
        llvm::IntegerType::get(*context_, decode->width());
    // If the input value is greater than this op's width, then return 0.
    // In that case, the shl will produce a poison value, but it'll be unused.
    llvm::Value* cast_input = builder_->CreateZExt(input, result_type);
    llvm::Value* overflow = builder_->CreateICmpUGE(
        cast_input, llvm::ConstantInt::get(result_type, decode->width()));
    llvm::Value* result = builder_->CreateSelect(
        overflow, llvm::ConstantInt::get(result_type, 0),
        builder_->CreateShl(llvm::ConstantInt::get(result_type, 1),
                            cast_input));

    return StoreResult(decode, result);
  }

  absl::Status HandleEncode(Encode* encode) override {
    llvm::Value* input = node_map_.at(encode->operand(0));
    llvm::Type* input_type = input->getType();
    llvm::Value* input_one = llvm::ConstantInt::get(input_type, 1);

    llvm::Type* result_type =
        type_converter_->ConvertToLlvmType(*encode->GetType());
    llvm::Value* result = llvm::ConstantInt::get(result_type, 0);

    llvm::Value* result_zero = llvm::ConstantInt::get(result_type, 0);

    // For each bit in the input, if it's set, bitwise-OR its [numeric] value
    // with the result.
    for (int i = 0; i < input_type->getIntegerBitWidth(); ++i) {
      llvm::Value* bit_set = builder_->CreateICmpEQ(
          builder_->CreateAnd(input, input_one), input_one);

      // Chained select, i.e., a = (b ? c : (d ? e : (...))), etc.
      llvm::Value* or_value = builder_->CreateSelect(
          bit_set, llvm::ConstantInt::get(result_type, i), result_zero);
      result = builder_->CreateOr(result, or_value);

      input = builder_->CreateLShr(input, input_one);
    }

    return StoreResult(encode, result);
  }

  absl::Status HandleEq(CompareOp* eq) override {
    llvm::Value* lhs = node_map_.at(eq->operand(0));
    llvm::Value* rhs = node_map_.at(eq->operand(1));
    llvm::Value* result = builder_->CreateICmpEQ(lhs, rhs);
    return StoreResult(eq, result);
  }

  absl::Status HandleIdentity(UnOp* identity) override {
    return StoreResult(identity, node_map_.at(identity->operand(0)));
  }

  absl::Status HandleInvoke(Invoke* invoke) override {
    XLS_ASSIGN_OR_RETURN(llvm::Function * function,
                         GetModuleFunction(invoke->to_apply()));

    std::vector<llvm::Value*> args(invoke->operand_count());
    for (int i = 0; i < invoke->operand_count(); i++) {
      args[i] = node_map_[invoke->operand(i)];
    }

    llvm::Value* invoke_inst = builder_->CreateCall(function, args);
    return StoreResult(invoke, invoke_inst);
  }

  absl::Status HandleLiteral(Literal* literal) override {
    Type* xls_type = literal->GetType();
    XLS_ASSIGN_OR_RETURN(
        llvm::Value * llvm_literal,
        type_converter_->ToLlvmConstant(*xls_type, literal->value()));

    return StoreResult(literal, llvm_literal);
  }

  absl::Status HandleMap(Map* map) override {
    XLS_ASSIGN_OR_RETURN(llvm::Function * to_apply,
                         GetModuleFunction(map->to_apply()));

    llvm::Value* input = node_map_.at(map->operand(0));
    llvm::Type* input_type = input->getType();
    llvm::FunctionType* function_type = llvm::cast<llvm::FunctionType>(
        to_apply->getType()->getPointerElementType());

    llvm::Value* result = CreateTypedZeroValue(llvm::ArrayType::get(
        function_type->getReturnType(), input_type->getArrayNumElements()));

    for (uint32 i = 0; i < input_type->getArrayNumElements(); ++i) {
      llvm::Value* iter_input = builder_->CreateExtractValue(input, {i});
      llvm::Value* iter_result = builder_->CreateCall(to_apply, iter_input);
      result = builder_->CreateInsertValue(result, iter_result, {i});
    }

    return StoreResult(map, result);
  }

  absl::Status HandleSMul(ArithOp* mul) override { return HandleArithOp(mul); }

  absl::Status HandleUMul(ArithOp* mul) override { return HandleArithOp(mul); }

  absl::Status HandleNaryAnd(NaryOp* and_op) override {
    llvm::Value* result = node_map_.at((and_op->operand(0)));
    for (int i = 1; i < and_op->operand_count(); ++i) {
      result = builder_->CreateAnd(result, node_map_.at(and_op->operand(i)));
    }
    return StoreResult(and_op, result);
  }

  absl::Status HandleNaryNand(NaryOp* nand_op) override {
    llvm::Value* result = node_map_.at((nand_op->operand(0)));
    for (int i = 1; i < nand_op->operand_count(); ++i) {
      result = builder_->CreateAnd(result, node_map_.at(nand_op->operand(i)));
    }
    result = builder_->CreateNot(result);
    return StoreResult(nand_op, result);
  }

  absl::Status HandleNaryNor(NaryOp* nor_op) override {
    llvm::Value* result = node_map_.at((nor_op->operand(0)));
    for (int i = 1; i < nor_op->operand_count(); ++i) {
      result = builder_->CreateOr(result, node_map_.at(nor_op->operand(i)));
    }
    result = builder_->CreateNot(result);
    return StoreResult(nor_op, result);
  }

  absl::Status HandleNaryOr(NaryOp* or_op) override {
    llvm::Value* result = node_map_.at((or_op->operand(0)));
    for (int i = 1; i < or_op->operand_count(); ++i) {
      result = builder_->CreateOr(result, node_map_.at(or_op->operand(i)));
    }
    return StoreResult(or_op, result);
  }

  absl::Status HandleNaryXor(NaryOp* xor_op) override {
    llvm::Value* result = node_map_.at((xor_op->operand(0)));
    for (int i = 1; i < xor_op->operand_count(); ++i) {
      result = builder_->CreateXor(result, node_map_.at(xor_op->operand(i)));
    }
    return StoreResult(xor_op, result);
  }

  absl::Status HandleNe(CompareOp* ne) override {
    llvm::Value* lhs = node_map_.at(ne->operand(0));
    llvm::Value* rhs = node_map_.at(ne->operand(1));
    llvm::Value* result = builder_->CreateICmpNE(lhs, rhs);
    return StoreResult(ne, result);
  }

  absl::Status HandleNeg(UnOp* neg) override {
    llvm::Value* llvm_neg = builder_->CreateNeg(node_map_.at(neg->operand(0)));
    return StoreResult(neg, llvm_neg);
  }

  absl::Status HandleNot(UnOp* not_op) override {
    llvm::Value* llvm_not =
        builder_->CreateNot(node_map_.at(not_op->operand(0)));
    return StoreResult(not_op, llvm_not);
  }

  absl::Status HandleOneHot(OneHot* one_hot) override {
    llvm::Value* input = node_map_.at(one_hot->operand(0));
    llvm::Type* input_type = input->getType();
    int input_width = input_type->getIntegerBitWidth();
    llvm::Type* int1_type = llvm::Type::getInt1Ty(*context_);
    std::vector<llvm::Type*> arg_types = {input_type, int1_type};
    llvm::Value* llvm_false = llvm::ConstantInt::getFalse(int1_type);

    llvm::Value* zeroes;
    if (one_hot->priority() == LsbOrMsb::kLsb) {
      llvm::Function* cttz = llvm::Intrinsic::getDeclaration(
          module_, llvm::Intrinsic::cttz, arg_types);
      zeroes = builder_->CreateCall(cttz, {input, llvm_false});
    } else {
      llvm::Function* ctlz = llvm::Intrinsic::getDeclaration(
          module_, llvm::Intrinsic::ctlz, arg_types);
      zeroes = builder_->CreateCall(ctlz, {input, llvm_false});
      zeroes = builder_->CreateSub(
          llvm::ConstantInt::get(input_type, input_width - 1), zeroes);
    }

    // If the input is zero, then return the special high-bit value.
    llvm::Value* zero_value = llvm::ConstantInt::get(input_type, 0);
    llvm::Value* width_value = llvm::ConstantInt::get(input_type, input_width);
    llvm::Value* eq_zero = builder_->CreateICmpEQ(input, zero_value);
    llvm::Value* shift_amount =
        builder_->CreateSelect(eq_zero, width_value, zeroes);

    llvm::Type* result_type = input_type->getWithNewBitWidth(input_width + 1);
    llvm::Value* result =
        builder_->CreateShl(llvm::ConstantInt::get(result_type, 1),
                            builder_->CreateZExt(shift_amount, result_type));
    return StoreResult(one_hot, result);
  }

  absl::Status HandleOneHotSel(OneHotSelect* sel) override {
    absl::Span<Node* const> cases = sel->cases();
    llvm::Type* input_type = node_map_.at(cases[0])->getType();

    llvm::Value* result;
    result = CreateTypedZeroValue(input_type);

    llvm::Value* selector = node_map_.at(sel->selector());
    llvm::Value* typed_zero = CreateTypedZeroValue(input_type);
    llvm::Value* llvm_one = llvm::ConstantInt::get(selector->getType(), 1);

    for (const auto* node : cases) {
      // Extract the current selector bit & see if set (CreateSelect requires an
      // i1 argument, or we could directly use the AND result.
      llvm::Value* is_hot = builder_->CreateICmpEQ(
          builder_->CreateAnd(selector, llvm_one), llvm_one);

      // OR with zero might be slower than doing an if/else construct - if
      // it turns out to be performance-critical, we can update it.
      llvm::Value* or_value =
          builder_->CreateSelect(is_hot, node_map_.at(node), typed_zero);
      result = CreateAggregateOr(result, or_value);
      selector = builder_->CreateLShr(selector, llvm_one);
    }

    return StoreResult(sel, result);
  }

  absl::Status HandleOrReduce(BitwiseReductionOp* op) override {
    // OR-reduce is equivalent to checking if any bit is set in the input.
    llvm::Value* operand = node_map_.at(op->operand(0));
    llvm::Value* eq = builder_->CreateICmpNE(
        operand, llvm::ConstantInt::get(operand->getType(), 0));
    return StoreResult(op, eq);
  }

  absl::Status HandleParam(Param* param) override {
    // If we're not processing the first function in LLVM space, this is easy -
    // just return the n'th argument to the active function.
    //
    // If this IS that entry function, then we need to pull in data from the
    // opaque arg buffer:
    //  1. Find out the index of the param we're loading.
    //  2. Get the offset of that param into our arg buffer.
    //  3. Cast that offset/pointer into the target type and load from it.
    XLS_ASSIGN_OR_RETURN(int index, param->function()->GetParamIndex(param));
    llvm::Function* llvm_function = builder_->GetInsertBlock()->getParent();

    if (!llvm_entry_function_ || param->function() != *llvm_entry_function_) {
      return StoreResult(param, llvm_function->getArg(index));
    }

    // Remember that all input arg pointers are packed into a buffer specified
    // as a single formal parameter, hence the 0 constant here.
    llvm::Argument* arg_pointer = llvm_function->getArg(0);

    llvm::Type* arg_type =
        type_converter_->ConvertToLlvmType(*param->GetType());
    llvm::Type* llvm_arg_ptr_type =
        llvm::PointerType::get(arg_type, /*AddressSpace=*/0);

    // Load 1: Get the pointer to arg N out of memory (the arg redirect buffer).
    llvm::Value* gep = builder_->CreateGEP(
        arg_pointer,
        {
            llvm::ConstantInt::get(llvm::Type::getInt64Ty(*context_), 0),
            llvm::ConstantInt::get(llvm::Type::getInt64Ty(*context_), index),
        });
    llvm::LoadInst* load =
        builder_->CreateLoad(gep->getType()->getPointerElementType(), gep);
    llvm::Value* cast = builder_->CreateBitCast(load, llvm_arg_ptr_type);

    // Load 2: Get the data at that pointer's destination.
    load = builder_->CreateLoad(arg_type, cast);

    return StoreResult(param, load);
  }

  absl::Status HandleReverse(UnOp* reverse) override {
    llvm::Value* input = node_map_.at(reverse->operand(0));
    llvm::Function* reverse_fn = llvm::Intrinsic::getDeclaration(
        module_, llvm::Intrinsic::bitreverse, {input->getType()});
    return StoreResult(reverse, builder_->CreateCall(reverse_fn, {input}));
  }

  absl::Status HandleSDiv(BinOp* binop) override { return HandleBinOp(binop); }

  absl::Status HandleSel(Select* sel) override {
    // Sel is implemented by a cascading series of select ops, e.g.,
    // selector == 0 ? cases[0] : selector == 1 ? cases[1] : selector == 2 ? ...
    llvm::Value* selector = node_map_.at(sel->selector());
    llvm::Value* llvm_sel =
        sel->default_value() ? node_map_.at(*sel->default_value()) : nullptr;
    for (int i = sel->cases().size() - 1; i >= 0; i--) {
      Node* node = sel->cases()[i];
      if (llvm_sel == nullptr) {
        // The last element in the select tree isn't a sel, but an actual value.
        llvm_sel = node_map_.at(node);
      } else {
        llvm::Value* index = llvm::ConstantInt::get(selector->getType(), i);
        llvm::Value* cmp = builder_->CreateICmpEQ(selector, index);
        llvm_sel = builder_->CreateSelect(cmp, node_map_.at(node), llvm_sel);
      }
    }
    return StoreResult(sel, llvm_sel);
  }

  absl::Status HandleSGe(CompareOp* ge) override {
    llvm::Value* lhs = node_map_.at(ge->operand(0));
    llvm::Value* rhs = node_map_.at(ge->operand(1));
    llvm::Value* result = builder_->CreateICmpSGE(lhs, rhs);
    return StoreResult(ge, result);
  }

  absl::Status HandleSGt(CompareOp* gt) override {
    llvm::Value* lhs = node_map_.at(gt->operand(0));
    llvm::Value* rhs = node_map_.at(gt->operand(1));
    llvm::Value* result = builder_->CreateICmpSGT(lhs, rhs);
    return StoreResult(gt, result);
  }

  absl::Status HandleSignExtend(ExtendOp* sign_ext) override {
    llvm::Type* new_type =
        llvm::IntegerType::get(*context_, sign_ext->new_bit_count());
    return StoreResult(
        sign_ext,
        builder_->CreateSExt(node_map_.at(sign_ext->operand(0)), new_type));
  }

  absl::Status HandleSLe(CompareOp* le) override {
    llvm::Value* lhs = node_map_.at(le->operand(0));
    llvm::Value* rhs = node_map_.at(le->operand(1));
    llvm::Value* result = builder_->CreateICmpSLE(lhs, rhs);
    return StoreResult(le, result);
  }

  absl::Status HandleSLt(CompareOp* lt) override {
    llvm::Value* lhs = node_map_.at(lt->operand(0));
    llvm::Value* rhs = node_map_.at(lt->operand(1));
    llvm::Value* result = builder_->CreateICmpSLT(lhs, rhs);
    return StoreResult(lt, result);
  }

  absl::Status HandleShll(BinOp* binop) override { return HandleBinOp(binop); }

  absl::Status HandleShra(BinOp* binop) override { return HandleBinOp(binop); }

  absl::Status HandleShrl(BinOp* binop) override { return HandleBinOp(binop); }

  absl::Status HandleSub(BinOp* binop) override { return HandleBinOp(binop); }

  absl::Status HandleTuple(Tuple* tuple) override {
    llvm::Type* tuple_type =
        type_converter_->ConvertToLlvmType(*tuple->GetType());

    llvm::Value* result = CreateTypedZeroValue(tuple_type);
    for (uint32 i = 0; i < tuple->operand_count(); ++i) {
      result = builder_->CreateInsertValue(
          result, node_map_.at(tuple->operand(i)), {i});
    }

    return StoreResult(tuple, result);
  }

  absl::Status HandleTupleIndex(TupleIndex* index) override {
    llvm::Value* value = builder_->CreateExtractValue(
        node_map_.at(index->operand(0)), index->index());
    return StoreResult(index, value);
  }

  absl::Status HandleUDiv(BinOp* binop) override { return HandleBinOp(binop); }

  absl::Status HandleUGe(CompareOp* ge) override {
    llvm::Value* lhs = node_map_.at(ge->operand(0));
    llvm::Value* rhs = node_map_.at(ge->operand(1));
    llvm::Value* result = builder_->CreateICmpUGE(lhs, rhs);
    return StoreResult(ge, result);
  }

  absl::Status HandleUGt(CompareOp* gt) override {
    llvm::Value* lhs = node_map_.at(gt->operand(0));
    llvm::Value* rhs = node_map_.at(gt->operand(1));
    llvm::Value* result = builder_->CreateICmpUGT(lhs, rhs);
    return StoreResult(gt, result);
  }

  absl::Status HandleULe(CompareOp* le) override {
    llvm::Value* lhs = node_map_.at(le->operand(0));
    llvm::Value* rhs = node_map_.at(le->operand(1));
    llvm::Value* result = builder_->CreateICmpULE(lhs, rhs);
    return StoreResult(le, result);
  }

  absl::Status HandleULt(CompareOp* lt) override {
    llvm::Value* lhs = node_map_.at(lt->operand(0));
    llvm::Value* rhs = node_map_.at(lt->operand(1));
    llvm::Value* result = builder_->CreateICmpULT(lhs, rhs);
    return StoreResult(lt, result);
  }

  absl::Status HandleXorReduce(BitwiseReductionOp* op) override {
    // XOR-reduce is equivalent to checking if the number of set bits is odd.
    llvm::Value* operand = node_map_.at(op->operand(0));
    llvm::Function* ctpop = llvm::Intrinsic::getDeclaration(
        module_, llvm::Intrinsic::ctpop, {operand->getType()});
    llvm::Value* pop_count = builder_->CreateCall(ctpop, {operand});

    // Once we have the pop count, truncate to the first (i.e., "is odd") bit.
    llvm::Value* truncated_value =
        builder_->CreateTrunc(pop_count, llvm::IntegerType::get(*context_, 1));
    return StoreResult(op, truncated_value);
  }

  absl::Status HandleZeroExtend(ExtendOp* zero_ext) override {
    llvm::Value* base = node_map_.at(zero_ext->operand(0));
    llvm::Type* dest_type =
        base->getType()->getWithNewBitWidth(zero_ext->new_bit_count());
    llvm::Value* zext =
        builder_->CreateZExt(node_map_.at(zero_ext->operand(0)), dest_type);
    return StoreResult(zero_ext, zext);
  }

  llvm::Value* return_value() { return return_value_; }

 private:
  absl::Status HandleArithOp(ArithOp* arith_op) {
    bool is_signed;
    switch (arith_op->op()) {
      case Op::kSMul:
        is_signed = true;
        break;
      case Op::kUMul:
        is_signed = false;
        break;
      default:
        return absl::InvalidArgumentError(absl::StrCat(
            "Unsupported arithmetic op:", OpToString(arith_op->op())));
    }
    llvm::Type* result_type =
        type_converter_->ConvertToLlvmType(*arith_op->GetType());
    llvm::Value* lhs = builder_->CreateIntCast(
        node_map_.at(arith_op->operands()[0]), result_type, is_signed);
    llvm::Value* rhs = builder_->CreateIntCast(
        node_map_.at(arith_op->operands()[1]), result_type, is_signed);

    llvm::Value* result;
    switch (arith_op->op()) {
      case Op::kUMul:
      case Op::kSMul:
        result = builder_->CreateMul(lhs, rhs);
        break;
      default:
        return absl::InvalidArgumentError(absl::StrCat(
            "Unsupported arithmetic op:", OpToString(arith_op->op())));
    }
    return StoreResult(arith_op, result);
  }

  absl::Status HandleBinOp(BinOp* binop) {
    if (binop->operand_count() != 2) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Expected 2 args to a binary op; instead got %d",
                          binop->operand_count()));
    }

    llvm::Value* lhs = node_map_.at(binop->operands()[0]);
    llvm::Value* rhs = node_map_.at(binop->operands()[1]);
    llvm::Value* result;
    switch (binop->op()) {
      case Op::kAdd:
        result = builder_->CreateAdd(lhs, rhs);
        break;
      case Op::kShll:
      case Op::kShra:
      case Op::kShrl:
        result = EmitShiftOp(binop->op(), lhs, rhs);
        break;
      case Op::kSub:
        result = builder_->CreateSub(lhs, rhs);
        break;
      case Op::kUDiv:
        result = EmitDiv(lhs, rhs, /*is_signed=*/false);
        break;
      case Op::kSDiv:
        result = EmitDiv(lhs, rhs, /*is_signed=*/true);
        break;
      default:
        return absl::UnimplementedError(
            absl::StrFormat("Unsupported/unimplemented bin op: %d",
                            static_cast<int>(binop->op())));
    }

    return StoreResult(binop, result);
  }

  llvm::Value* EmitShiftOp(Op op, llvm::Value* lhs, llvm::Value* rhs) {
    // Shift operands are allowed to be different sizes in the [XLS] IR, so
    // we need to cast them to be the same size here (for LLVM).
    int common_width = std::max(lhs->getType()->getIntegerBitWidth(),
                                rhs->getType()->getIntegerBitWidth());
    llvm::Type* dest_type = llvm::IntegerType::get(*context_, common_width);
    lhs = builder_->CreateZExt(lhs, dest_type);
    rhs = builder_->CreateZExt(rhs, dest_type);
    // In LLVM, shift overflow creates poison. In XLS, it creates zero.
    llvm::Value* overflows = builder_->CreateICmpUGE(
        rhs, llvm::ConstantInt::get(dest_type, common_width));

    llvm::Value* inst;
    llvm::Value* zero = llvm::ConstantInt::get(dest_type, 0);
    llvm::Value* overflow_value = zero;
    if (op == Op::kShll) {
      inst = builder_->CreateShl(lhs, rhs);
    } else if (op == Op::kShra) {
      llvm::Value* high_bit = builder_->CreateLShr(
          lhs, llvm::ConstantInt::get(
                   dest_type, lhs->getType()->getIntegerBitWidth() - 1));
      llvm::Value* high_bit_set = builder_->CreateICmpEQ(
          high_bit, llvm::ConstantInt::get(dest_type, 1));
      overflow_value = builder_->CreateSelect(
          high_bit_set, llvm::ConstantInt::getSigned(dest_type, -1), zero);
      inst = builder_->CreateAShr(lhs, rhs);
    } else {
      inst = builder_->CreateLShr(lhs, rhs);
    }
    return builder_->CreateSelect(overflows, overflow_value, inst);
  }

  llvm::Value* EmitDiv(llvm::Value* lhs, llvm::Value* rhs, bool is_signed) {
    // XLS div semantics differ from LLVM's (and most software's) here: in XLS,
    // division by zero returns the greatest value of that type, so 255 for an
    // unsigned byte, and either -128 or 127 for a signed one.
    // Thus, a little more work is necessary to emit LLVM IR matching the XLS
    // div op than just IRBuilder::Create[SU]Div().
    int type_width = rhs->getType()->getIntegerBitWidth();
    llvm::Value* zero = llvm::ConstantInt::get(rhs->getType(), 0);
    llvm::Value* rhs_eq_zero = builder_->CreateICmpEQ(rhs, zero);
    llvm::Value* lhs_gt_zero = builder_->CreateICmpSGT(lhs, zero);

    // If rhs is zero, make LHS = the max/min value and the RHS 1,
    // rather than introducing a proper conditional.
    rhs = builder_->CreateSelect(
        rhs_eq_zero, llvm::ConstantInt::get(rhs->getType(), 1), rhs);
    if (is_signed) {
      llvm::Value* max_value =
          type_converter_
              ->ToLlvmConstant(rhs->getType(),
                               Value(Bits::MaxSigned(type_width)))
              .value();
      llvm::Value* min_value =
          type_converter_
              ->ToLlvmConstant(rhs->getType(),
                               Value(Bits::MinSigned(type_width)))
              .value();

      lhs = builder_->CreateSelect(
          rhs_eq_zero,
          builder_->CreateSelect(lhs_gt_zero, max_value, min_value), lhs);
      return builder_->CreateSDiv(lhs, rhs);
    }

    lhs = builder_->CreateSelect(
        rhs_eq_zero,
        type_converter_
            ->ToLlvmConstant(rhs->getType(), Value(Bits::AllOnes(type_width)))
            .value(),
        lhs);
    return builder_->CreateUDiv(lhs, rhs);
  }

  llvm::Constant* CreateTypedZeroValue(llvm::Type* type) {
    if (type->isIntegerTy()) {
      return llvm::ConstantInt::get(type, 0);
    } else if (type->isArrayTy()) {
      std::vector<llvm::Constant*> elements(
          type->getArrayNumElements(),
          CreateTypedZeroValue(type->getArrayElementType()));
      return llvm::ConstantArray::get(llvm::cast<llvm::ArrayType>(type),
                                      elements);
    }

    // Must be a tuple/struct, then.
    std::vector<llvm::Constant*> elements(type->getStructNumElements());
    for (int i = 0; i < type->getStructNumElements(); ++i) {
      elements[i] = CreateTypedZeroValue(type->getStructElementType(i));
    }

    return llvm::ConstantStruct::get(llvm::cast<llvm::StructType>(type),
                                     elements);
  }

  llvm::Value* CreateAggregateOr(llvm::Value* lhs, llvm::Value* rhs) {
    llvm::Type* arg_type = lhs->getType();
    if (arg_type->isIntegerTy()) {
      return builder_->CreateOr(lhs, rhs);
    }

    llvm::Value* result = CreateTypedZeroValue(arg_type);
    int num_elements = arg_type->isArrayTy() ? arg_type->getArrayNumElements()
                                             : arg_type->getNumContainedTypes();
    for (uint32 i = 0; i < num_elements; ++i) {
      llvm::Value* iter_result =
          CreateAggregateOr(builder_->CreateExtractValue(lhs, {i}),
                            builder_->CreateExtractValue(rhs, {i}));
      result = builder_->CreateInsertValue(result, iter_result, {i});
    }

    return result;
  }

  xabsl::StatusOr<llvm::Constant*> ConvertToLlvmConstant(Type* type,
                                                         const Value& value) {
    if (type->IsBits()) {
      return type_converter_->ToLlvmConstant(
          type_converter_->ConvertToLlvmType(*type), value);
    } else if (type->IsTuple()) {
      TupleType* tuple_type = type->AsTupleOrDie();
      std::vector<llvm::Constant*> llvm_elements;
      for (int i = 0; i < tuple_type->size(); ++i) {
        XLS_ASSIGN_OR_RETURN(
            llvm::Constant * llvm_element,
            type_converter_->ToLlvmConstant(*tuple_type->element_type(i),
                                            value.element(i)));
        llvm_elements.push_back(llvm_element);
      }

      llvm::Type* llvm_type = type_converter_->ConvertToLlvmType(*type);
      return llvm::ConstantStruct::get(llvm::cast<llvm::StructType>(llvm_type),
                                       llvm_elements);
    } else if (type->IsArray()) {
      const ArrayType* array_type = type->AsArrayOrDie();
      std::vector<llvm::Constant*> elements;
      for (const Value& element : value.elements()) {
        XLS_ASSIGN_OR_RETURN(llvm::Constant * llvm_element,
                             type_converter_->ToLlvmConstant(
                                 *array_type->element_type(), element));
        elements.push_back(llvm_element);
      }

      llvm::Type* element_type =
          type_converter_->ConvertToLlvmType(*array_type->element_type());
      return llvm::ConstantArray::get(
          llvm::ArrayType::get(element_type, array_type->size()), elements);
    }

    XLS_LOG(FATAL) << "Unknown value kind: " << value.kind();
  }

  xabsl::StatusOr<llvm::Function*> GetModuleFunction(Function* xls_function) {
    // If we've not processed this function yet, then do so.
    llvm::Function* found_function = module_->getFunction(xls_function->name());
    if (found_function != nullptr) {
      return found_function;
    }

    // There are a couple of differences between this and entry function
    // visitor initialization such that I think it makes slightly more sense
    // to not factor it into a common block, but it's not clear-cut.
    std::vector<llvm::Type*> param_types(xls_function->params().size());
    for (int i = 0; i < xls_function->params().size(); ++i) {
      param_types[i] = type_converter_->ConvertToLlvmType(
          *xls_function->param(i)->GetType());
    }

    Type* return_type = xls_function->return_value()->GetType();
    llvm::Type* llvm_return_type =
        type_converter_->ConvertToLlvmType(*return_type);

    llvm::FunctionType* function_type = llvm::FunctionType::get(
        llvm_return_type,
        llvm::ArrayRef<llvm::Type*>(param_types.data(), param_types.size()),
        /*isVarArg=*/false);
    llvm::Function* function = llvm::cast<llvm::Function>(
        module_
            ->getOrInsertFunction(xls_function->qualified_name(), function_type)
            .getCallee());

    llvm::BasicBlock* block =
        llvm::BasicBlock::Create(*context_, xls_function->qualified_name(),
                                 function, /*InsertBefore=*/nullptr);
    llvm::IRBuilder<> builder(block);
    BuilderVisitor visitor(module_, &builder, {}, absl::nullopt,
                           type_converter_);
    XLS_RETURN_IF_ERROR(xls_function->Accept(&visitor));
    if (function_type->getReturnType()->isVoidTy()) {
      builder.CreateRetVoid();
    } else {
      builder.CreateRet(visitor.return_value());
    }

    return function;
  }

  absl::Status StoreResult(Node* node, llvm::Value* value) {
    XLS_RET_CHECK(!node_map_.contains(node));
    value->setName(verilog::SanitizeIdentifier(node->GetName()));
    if (node->function()->return_value() == node) {
      return_value_ = value;
    }
    node_map_[node] = value;

    return absl::OkStatus();
  }

  llvm::Module* module_;
  llvm::LLVMContext* context_;
  llvm::IRBuilder<>* builder_;

  // List of start/end (inclusive) byte offset pairs for each argument to this
  // visitor's function.
  std::vector<std::pair<int64, int64>> arg_indices_;

  // The last value constructed during this traversal - represents the return
  // from calculation.
  llvm::Value* return_value_;

  // Maps an XLS Node to the resulting LLVM Value.
  absl::flat_hash_map<Node*, llvm::Value*> node_map_;

  // Holds storage for array indexing ops. Since we need to dump arrays to
  // storage to extract elements (i.e., for GEPs), it makes sense to only create
  // and store the array once.
  absl::flat_hash_map<llvm::Value*, llvm::AllocaInst*> array_storage_;

  LlvmTypeConverter* type_converter_;

  // The entry point into LLVM space - the function specified in the constructor
  // to the top-level LlvmIrJit object.
  absl::optional<Function*> llvm_entry_function_;
};

absl::once_flag once;
void OnceInit() {
  LLVMInitializeNativeTarget();
  LLVMInitializeNativeAsmPrinter();
  LLVMInitializeNativeAsmParser();
}

}  // namespace

xabsl::StatusOr<std::unique_ptr<LlvmIrJit>> LlvmIrJit::Create(
    Function* xls_function, int64 opt_level) {
  absl::call_once(once, OnceInit);

  auto jit = absl::WrapUnique(new LlvmIrJit(xls_function, opt_level));
  XLS_RETURN_IF_ERROR(jit->Init());
  XLS_RETURN_IF_ERROR(jit->CompileFunction());
  return jit;
}

LlvmIrJit::LlvmIrJit(Function* xls_function, int64 opt_level)
    : context_(std::make_unique<llvm::LLVMContext>()),
      object_layer_(
          execution_session_,
          []() { return std::make_unique<llvm::SectionMemoryManager>(); }),
      dylib_(execution_session_.createBareJITDylib("main")),
      data_layout_(""),
      xls_function_(xls_function),
      xls_function_type_(xls_function_->GetType()),
      opt_level_(opt_level),
      invoker_(nullptr) {}

llvm::Expected<llvm::orc::ThreadSafeModule> LlvmIrJit::Optimizer(
    llvm::orc::ThreadSafeModule module,
    const llvm::orc::MaterializationResponsibility& responsibility) {
  llvm::Module* bare_module = module.getModuleUnlocked();

  XLS_VLOG(2) << "Unoptimized module IR:";
  XLS_VLOG(2).NoPrefix() << ir_runtime_->DumpToString(*bare_module);

  llvm::TargetLibraryInfoImpl library_info(target_machine_->getTargetTriple());
  llvm::PassManagerBuilder builder;
  builder.OptLevel = opt_level_;
  builder.LibraryInfo =
      new llvm::TargetLibraryInfoImpl(target_machine_->getTargetTriple());

  llvm::legacy::PassManager module_pass_manager;
  builder.populateModulePassManager(module_pass_manager);
  module_pass_manager.add(llvm::createTargetTransformInfoWrapperPass(
      target_machine_->getTargetIRAnalysis()));

  llvm::legacy::FunctionPassManager function_pass_manager(bare_module);
  builder.populateFunctionPassManager(function_pass_manager);
  function_pass_manager.doInitialization();
  for (auto& function : *bare_module) {
    function_pass_manager.run(function);
  }
  function_pass_manager.doFinalization();

  bool dump_asm = false;
  llvm::SmallVector<char, 0> stream_buffer;
  llvm::raw_svector_ostream ostream(stream_buffer);
  if (XLS_VLOG_IS_ON(3)) {
    dump_asm = true;
    if (target_machine_->addPassesToEmitFile(
            module_pass_manager, ostream, nullptr, llvm::CGFT_AssemblyFile)) {
      XLS_VLOG(3) << "Could not create ASM generation pass!";
      dump_asm = false;
    }
  }

  module_pass_manager.run(*bare_module);

  XLS_VLOG(2) << "Optimized module IR:";
  XLS_VLOG(2).NoPrefix() << ir_runtime_->DumpToString(*bare_module);

  if (dump_asm) {
    XLS_VLOG(3) << "Generated ASM:";
    XLS_VLOG_LINES(3, std::string(stream_buffer.begin(), stream_buffer.end()));
  }
  return module;
}

absl::Status LlvmIrJit::Init() {
  auto error_or_target_builder =
      llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!error_or_target_builder) {
    return absl::InternalError(
        absl::StrCat("Unable to detect host: ",
                     llvm::toString(error_or_target_builder.takeError())));
  }

  auto error_or_target_machine = error_or_target_builder->createTargetMachine();
  if (!error_or_target_machine) {
    return absl::InternalError(
        absl::StrCat("Unable to create target machine: ",
                     llvm::toString(error_or_target_machine.takeError())));
  }
  target_machine_ = std::move(error_or_target_machine.get());
  data_layout_ = target_machine_->createDataLayout();

  execution_session_.runSessionLocked([this]() {
    dylib_.addGenerator(
        cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            data_layout_.getGlobalPrefix())));
  });

  auto compiler = std::make_unique<llvm::orc::SimpleCompiler>(*target_machine_);
  compile_layer_ = std::make_unique<llvm::orc::IRCompileLayer>(
      execution_session_, object_layer_, std::move(compiler));

  transform_layer_ = std::make_unique<llvm::orc::IRTransformLayer>(
      execution_session_, *compile_layer_,
      [this](llvm::orc::ThreadSafeModule module,
             const llvm::orc::MaterializationResponsibility& responsibility) {
        return Optimizer(std::move(module), responsibility);
      });

  return absl::OkStatus();
}

absl::Status LlvmIrJit::CompileFunction() {
  llvm::LLVMContext* bare_context = context_.getContext();
  auto module = std::make_unique<llvm::Module>("the_module", *bare_context);
  module->setDataLayout(data_layout_);
  type_converter_ =
      std::make_unique<LlvmTypeConverter>(bare_context, data_layout_);

  // To return values > 64b in size, we need to copy them into a result buffer,
  // instead of returning a fixed-size result element.
  // To do this, we need to construct the function type, adding a result buffer
  // arg (and setting the result type to void) and then storing the computation
  // result therein.
  std::vector<llvm::Type*> param_types;
  llvm::FunctionType* function_type;
  // Create a dummy param to hold a packed representation of the input args.
  param_types.push_back(llvm::PointerType::get(
      llvm::ArrayType::get(
          llvm::PointerType::get(llvm::Type::getInt8Ty(*bare_context),
                                 /*AddressSpace=*/0),
          xls_function_type_->parameter_count()),
      /*AddressSpace=*/0));

  for (const Type* type : xls_function_type_->parameters()) {
    arg_type_bytes_.push_back(type_converter_->GetTypeByteSize(*type));
  }

  // Since we only pass around concrete values (i.e., not functions), we can
  // use the flat byte count of the XLS type to size our result array.
  Type* return_type = xls_function_type_->return_type();
  llvm::Type* llvm_return_type =
      type_converter_->ConvertToLlvmType(*return_type);
  param_types.push_back(
      llvm::PointerType::get(llvm_return_type, /*AddressSpace=*/0));
  function_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(*bare_context),
      llvm::ArrayRef<llvm::Type*>(param_types.data(), param_types.size()),
      /*isVarArg=*/false);

  Package* xls_package = xls_function_->package();
  std::string function_name =
      absl::StrFormat("%s::%s", xls_package->name(), xls_function_->name());
  llvm::Function* llvm_function = llvm::cast<llvm::Function>(
      module->getOrInsertFunction(function_name, function_type).getCallee());
  auto basic_block = llvm::BasicBlock::Create(
      *bare_context, "so_basic", llvm_function, /*InsertBefore=*/nullptr);

  llvm::IRBuilder<> builder(basic_block);
  BuilderVisitor visitor(module.get(), &builder, xls_function_->params(),
                         xls_function_, type_converter_.get());
  XLS_RETURN_IF_ERROR(xls_function_->Accept(&visitor));
  llvm::Value* return_value = visitor.return_value();
  if (return_value == nullptr) {
    return absl::InvalidArgumentError(
        "Function had no (or an unsupported) return value specification!");
  }

  // Store the result to the output pointer.
  return_type_bytes_ = type_converter_->GetTypeByteSize(*return_type);
  if (return_value->getType()->isPointerTy()) {
    llvm::Type* pointee_type = return_value->getType()->getPointerElementType();
    if (pointee_type != llvm_return_type) {
      std::string output;
      llvm::raw_string_ostream stream(output);
      stream << "Produced return type does not match intended: produced: ";
      pointee_type->print(stream, /*IsForDebug=*/true);
      stream << ", expected: ";
      llvm_return_type->print(stream, /*IsForDebug=*/true);
      return absl::InternalError(stream.str());
    }

    builder.CreateMemCpy(llvm_function->getArg(llvm_function->arg_size() - 1),
                         llvm::MaybeAlign(0), return_value, llvm::MaybeAlign(0),
                         return_type_bytes_);
  } else {
    builder.CreateStore(return_value,
                        llvm_function->getArg(llvm_function->arg_size() - 1));
  }
  builder.CreateRetVoid();

  llvm::Error error = transform_layer_->add(
      dylib_, llvm::orc::ThreadSafeModule(std::move(module), context_));
  if (error) {
    return absl::UnknownError(absl::StrFormat(
        "Error compiling converted IR: %s", llvm::toString(std::move(error))));
  }

  llvm::Expected<llvm::JITEvaluatedSymbol> symbol =
      execution_session_.lookup(&dylib_, function_name);
  if (!symbol) {
    return absl::InternalError(
        absl::StrFormat("Could not find start symbol \"%s\": %s", function_name,
                        llvm::toString(symbol.takeError())));
  }

  llvm::JITTargetAddress invoker = symbol->getAddress();
  invoker_ = reinterpret_cast<JitFunctionType>(invoker);

  ir_runtime_ =
      std::make_unique<LlvmIrRuntime>(data_layout_, type_converter_.get());

  return absl::OkStatus();
}

xabsl::StatusOr<Value> LlvmIrJit::Run(absl::Span<const Value> args) {
  absl::Span<Param* const> params = xls_function_->params();
  if (args.size() != params.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Arg list has the wrong size: %d vs expected %d.",
                        args.size(), xls_function_->params().size()));
  }

  for (int i = 0; i < params.size(); i++) {
    if (!ValueConformsToType(args[i], params[i]->GetType())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Got argument %s for parameter %d which is not of type %s",
          args[i].ToString(), i, params[i]->GetType()->ToString()));
    }
  }

  std::vector<std::unique_ptr<uint8[]>> unique_arg_buffers;
  std::vector<uint8*> arg_buffers;
  unique_arg_buffers.reserve(xls_function_type_->parameters().size());
  arg_buffers.reserve(unique_arg_buffers.size());
  for (const Type* type : xls_function_type_->parameters()) {
    unique_arg_buffers.push_back(
        std::make_unique<uint8[]>(type_converter_->GetTypeByteSize(*type)));
    arg_buffers.push_back(unique_arg_buffers.back().get());
  }

  XLS_RETURN_IF_ERROR(ir_runtime_->PackArgs(
      args, xls_function_type_->parameters(), absl::MakeSpan(arg_buffers)));

  absl::InlinedVector<uint8, 16> outputs(return_type_bytes_);
  invoker_(arg_buffers.data(), outputs.data());

  return ir_runtime_->UnpackBuffer(outputs.data(),
                                   xls_function_type_->return_type());
}

xabsl::StatusOr<Value> LlvmIrJit::Run(
    const absl::flat_hash_map<std::string, Value>& kwargs) {
  XLS_ASSIGN_OR_RETURN(std::vector<Value> positional_args,
                       KeywordArgsToPositional(*xls_function_, kwargs));
  return Run(positional_args);
}

absl::Status LlvmIrJit::RunWithViews(absl::Span<const uint8*> args,
                                     absl::Span<uint8> result_buffer) {
  absl::Span<Param* const> params = xls_function_->params();
  if (args.size() != params.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Arg list has the wrong size: %d vs expected %d.",
                        args.size(), xls_function_->params().size()));
  }

  if (result_buffer.size() < return_type_bytes_) {
    return absl::InvalidArgumentError(
        absl::StrCat("Result buffer too small - must be at least %d bytes!",
                     return_type_bytes_));
  }

  invoker_(args.data(), result_buffer.data());
  return absl::OkStatus();
}

xabsl::StatusOr<Value> CreateAndRun(Function* xls_function,
                                    absl::Span<const Value> args) {
  XLS_ASSIGN_OR_RETURN(auto jit, LlvmIrJit::Create(xls_function));
  XLS_ASSIGN_OR_RETURN(auto result, jit->Run(args));
  return result;
}

}  // namespace xls
