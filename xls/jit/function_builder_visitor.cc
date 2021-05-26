// Copyright 2020 The XLS Authors
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
#include "xls/jit/function_builder_visitor.h"

#include "llvm/include/llvm/IR/DerivedTypes.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/bits_ops.h"

#ifdef ABSL_HAVE_MEMORY_SANITIZER
#include <sanitizer/msan_interface.h>
#endif

#include "llvm/include/llvm/IR/Constants.h"
#include "xls/codegen/vast.h"
#include "xls/ir/function.h"
#include "xls/ir/proc.h"

namespace xls {

absl::Status FunctionBuilderVisitor::Visit(llvm::Module* module,
                                           llvm::Function* llvm_fn,
                                           FunctionBase* xls_fn,
                                           LlvmTypeConverter* type_converter,
                                           bool is_top, bool generate_packed) {
  XLS_VLOG_LINES(3, std::string("Generating LLVM IR for XLS function/proc:\n") +
                        xls_fn->DumpIr());
  FunctionBuilderVisitor visitor(module, llvm_fn, xls_fn, type_converter,
                                 is_top, generate_packed);
  return visitor.BuildInternal();
}

FunctionBuilderVisitor::FunctionBuilderVisitor(
    llvm::Module* module, llvm::Function* llvm_fn, FunctionBase* xls_fn,
    LlvmTypeConverter* type_converter, bool is_top, bool generate_packed)
    : ctx_(module->getContext()),
      module_(module),
      llvm_fn_(llvm_fn),
      xls_fn_(xls_fn),
      type_converter_(type_converter),
      is_top_(is_top),
      generate_packed_(generate_packed) {}

absl::Status FunctionBuilderVisitor::BuildInternal() {
  auto basic_block = llvm::BasicBlock::Create(ctx_, "so_basic", llvm_fn_,
                                              /*InsertBefore=*/nullptr);

  builder_ = std::make_unique<llvm::IRBuilder<>>(basic_block);
  XLS_RETURN_IF_ERROR(xls_fn_->Accept(this));
  if (return_value_ == nullptr) {
    return absl::InvalidArgumentError(
        "Function had no (or an unsupported) return value specification!");
  }

  // Store the result to the output pointer.
  Type* xls_return_type = GetEffectiveReturnValue(xls_fn_)->GetType();
  llvm::Type* llvm_return_type =
      type_converter_->ConvertToLlvmType(xls_return_type);
  if (!is_top_) {
    if (llvm_return_type->isVoidTy()) {
      builder_->CreateRetVoid();
    } else {
      builder_->CreateRet(return_value_);
    }
    return absl::OkStatus();
  }

  UnpoisonOutputBuffer();

  int64_t return_width = xls_return_type->GetFlatBitCount();
  llvm::Value* output_arg = GetOutputPtr();
  if (generate_packed_) {
    if (return_width != 0) {
      // Declare the return argument as an iX, and pack the actual data as such
      // an integer.
      llvm::Value* packed_return =
          llvm::ConstantInt::get(llvm::IntegerType::get(ctx_, return_width), 0);
      XLS_ASSIGN_OR_RETURN(
          packed_return,
          PackElement(return_value_, xls_return_type, packed_return, 0));
      builder_->CreateStore(packed_return, output_arg);
    }
    builder_->CreateRetVoid();
    return absl::OkStatus();
  }

  if (return_value_->getType()->isPointerTy()) {
    llvm::Type* pointee_type =
        return_value_->getType()->getPointerElementType();
    if (pointee_type != llvm_return_type) {
      std::string output;
      llvm::raw_string_ostream stream(output);
      stream << "Produced return type does not match intended: produced: ";
      pointee_type->print(stream, /*IsForDebug=*/true);
      stream << ", expected: ";
      llvm_return_type->print(stream, /*IsForDebug=*/true);
      return absl::InternalError(stream.str());
    }

    int64_t return_type_bytes =
        type_converter_->GetTypeByteSize(xls_return_type);
    builder_->CreateMemCpy(output_arg, llvm::MaybeAlign(0), return_value_,
                           llvm::MaybeAlign(0), return_type_bytes);
  } else {
    builder_->CreateStore(return_value_, output_arg);
  }
  builder_->CreateRetVoid();

  return absl::OkStatus();
}

absl::Status FunctionBuilderVisitor::HandleAdd(BinOp* binop) {
  return HandleBinOp(binop);
}

absl::Status FunctionBuilderVisitor::HandleAndReduce(BitwiseReductionOp* op) {
  // AND-reduce is equivalent to checking if every bit is set in the input.
  llvm::Value* operand = node_map_.at(op->operand(0));
  llvm::IntegerType* operand_type =
      llvm::cast<llvm::IntegerType>(operand->getType());
  llvm::Value* eq = builder_->CreateICmpEQ(
      operand, llvm::ConstantInt::get(operand_type, operand_type->getMask()));
  return StoreResult(op, eq);
}

absl::Status FunctionBuilderVisitor::HandleAfterAll(AfterAll* after_all) {
  // AfterAll is only meaningful to the compiler and does not actually perform
  // any computation. Furter, token types don't contain any data. A 0-element
  // array is a convenient and low-overhead way to let the rest of the llvm
  // infrastructure treat token like a normal data-type.
  return StoreResult(after_all, type_converter_->GetToken());
}

void RecordAssertion(char* msg, absl::Status* assert_status) {
  // Don't clobber a previously-recorded assertion failure.
  if (assert_status->ok()) {
    *assert_status = absl::AbortedError(msg);
  }
}

absl::Status FunctionBuilderVisitor::InvokeAssertCallback(
    llvm::IRBuilder<>* builder, const std::string& message) {
  llvm::Constant* msg_constant = builder->CreateGlobalStringPtr(message);

  llvm::Type* msg_type = msg_constant->getType();

  // Treat void pointers as int64_t values at the LLVM IR level.
  // Using an actual pointer type triggers LLVM asserts when compiling
  // in debug mode.
  // TODO(amfv): 2021-04-05 Figure out why and fix void pointer handling.
  llvm::Type* void_ptr_type = llvm::Type::getInt64Ty(ctx());

  std::vector<llvm::Type*> params = {msg_type, void_ptr_type};

  llvm::Type* void_type = llvm::Type::getVoidTy(ctx());

  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(void_type, params, /*isVarArg=*/false);

  llvm::Value* assert_status_ptr = GetAssertStatusPtr();

  std::vector<llvm::Value*> args = {msg_constant, assert_status_ptr};

  llvm::ConstantInt* fn_addr =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx()),
                             absl::bit_cast<uint64_t>(&RecordAssertion));
  llvm::Value* fn_ptr =
      builder->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));
  builder->CreateCall(fn_type, fn_ptr, args);
  return absl::OkStatus();
}

absl::Status FunctionBuilderVisitor::HandleAssert(Assert* assert_op) {
  std::string assert_label = assert_op->label().value_or("assert");

  llvm::BasicBlock* after_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(assert_label, "_after"), llvm_fn());

  llvm::BasicBlock* ok_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(assert_label, "_ok"), llvm_fn());

  llvm::IRBuilder<> ok_builder(ok_block);

  ok_builder.CreateBr(after_block);

  llvm::BasicBlock* fail_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(assert_label, "_fail"), llvm_fn());
  llvm::IRBuilder<> fail_builder(fail_block);
  XLS_RETURN_IF_ERROR(
      InvokeAssertCallback(&fail_builder, assert_op->message()));

  fail_builder.CreateBr(after_block);

  builder()->CreateCondBr(node_map().at(assert_op->condition()), ok_block,
                          fail_block);

  auto after_builder = std::make_unique<llvm::IRBuilder<>>(after_block);

  set_builder(std::move(after_builder));

  llvm::Value* token = type_converter_->GetToken();

  return StoreResult(assert_op, token);
}

absl::Status FunctionBuilderVisitor::HandleArray(Array* array) {
  llvm::Type* array_type = type_converter_->ConvertToLlvmType(array->GetType());

  llvm::Value* result = CreateTypedZeroValue(array_type);
  for (uint32_t i = 0; i < array->size(); ++i) {
    result = builder_->CreateInsertValue(result,
                                         node_map_.at(array->operand(i)), {i});
  }

  return StoreResult(array, result);
}

absl::StatusOr<llvm::Value*> FunctionBuilderVisitor::IndexIntoArray(
    llvm::Value* array, llvm::Value* index, int64_t array_size) {
  int64_t index_width = index->getType()->getIntegerBitWidth();

  // Check for out-of-bounds access. If the index is out of bounds it is set to
  // the maximum index value.
  int64_t index_bitwidth = index->getType()->getIntegerBitWidth();
  int64_t comparison_bitwidth = std::max(index_bitwidth, int64_t{64});
  llvm::Value* array_size_comparison_bitwidth = llvm::ConstantInt::get(
      llvm::Type::getIntNTy(ctx_, comparison_bitwidth), array_size);
  llvm::Value* index_value_comparison_bitwidth = builder_->CreateZExt(
      index, llvm::Type::getIntNTy(ctx_, comparison_bitwidth));
  llvm::Value* is_index_inbounds = builder_->CreateICmpULT(
      index_value_comparison_bitwidth, array_size_comparison_bitwidth);
  llvm::Value* inbounds_index = builder_->CreateSelect(
      is_index_inbounds, index,
      llvm::ConstantInt::get(index->getType(), array_size - 1));

  // Our IR does not use negative indices, so we add a
  // zero MSb to prevent LLVM from interpreting this as such.
  std::vector<llvm::Value*> gep_indices = {
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx_), 0),
      builder_->CreateZExt(inbounds_index,
                           llvm::IntegerType::get(ctx_, index_width + 1))};

  llvm::Type* array_type = array->getType();
  // Ideally, we'd use IRBuilder::CreateExtractValue here, but that requires
  // constant indices. Since there's no other way to extract a value from an
  // aggregate, we're left with storing the value in a temporary alloca and
  // using that pointer to extract the value.
  llvm::AllocaInst* alloca;
  if (!array_storage_.contains(array)) {
    alloca = builder_->CreateAlloca(array_type);
    builder_->CreateStore(array, alloca);
    array_storage_[array] = alloca;
  } else {
    alloca = array_storage_[array];
  }

  llvm::Type* element_type = array_type->getArrayElementType();
  llvm::Value* gep = builder_->CreateGEP(array_type, alloca, gep_indices);
  return builder_->CreateLoad(element_type, gep);
}

absl::Status FunctionBuilderVisitor::HandleArrayIndex(ArrayIndex* index) {
  Type* element_type = index->array()->GetType();
  llvm::Value* element = node_map_.at(index->array());
  for (Node* index_operand : index->indices()) {
    llvm::Value* index_value = node_map_.at(index_operand);
    XLS_ASSIGN_OR_RETURN(element,
                         IndexIntoArray(element, index_value,
                                        element_type->AsArrayOrDie()->size()));
    element_type = element_type->AsArrayOrDie()->element_type();
  }
  return StoreResult(index, element);
}

absl::Status FunctionBuilderVisitor::HandleArraySlice(ArraySlice* slice) {
  llvm::Value* array = node_map_.at(slice->array());
  llvm::Value* start = node_map_.at(slice->start());
  int64_t width = slice->width();

  // This overestimates the number of bits needed but in all practical
  // situations it should be fine. The only exception to that is if some code
  // uses a 64 bit index but doesn't actually make full use of that range, then
  // this will possibly push us over a performance cliff.
  int64_t index_bits = start->getType()->getIntegerBitWidth() +
                       Bits::MinBitCountSigned(width) + 1;
  llvm::Type* index_type = builder_->getIntNTy(index_bits);
  llvm::Type* result_type =
      type_converter_->ConvertToLlvmType(slice->GetType());
  llvm::Type* result_element_type = type_converter_->ConvertToLlvmType(
      slice->GetType()->AsArrayOrDie()->element_type());
  llvm::AllocaInst* alloca_uncasted = builder_->CreateAlloca(result_type);
  llvm::Value* alloca = builder_->CreateBitCast(
      alloca_uncasted,
      llvm::PointerType::get(result_element_type, /*AddressSpace=*/0),
      "alloca");
  llvm::Value* start_big = builder_->CreateZExt(start, index_type, "start_big");

  for (int64_t i = 0; i < width; i++) {
    llvm::Value* index = builder_->CreateAdd(
        start_big, llvm::ConstantInt::get(index_type, i), "index");
    XLS_ASSIGN_OR_RETURN(
        llvm::Value * value,
        IndexIntoArray(array, index,
                       slice->array()->GetType()->AsArrayOrDie()->size()));
    std::vector<llvm::Value*> gep_indices = {
        llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx_), i)};
    llvm::Value* gep =
        builder_->CreateGEP(result_element_type, alloca, gep_indices);
    builder_->CreateStore(value, gep);
  }

  llvm::Value* sliced_array =
      builder_->CreateLoad(result_type, alloca_uncasted);
  array_storage_[sliced_array] = alloca_uncasted;
  return StoreResult(slice, sliced_array);
}

absl::Status FunctionBuilderVisitor::HandleArrayUpdate(ArrayUpdate* update) {
  if (update->indices().empty()) {
    // An empty index replaces the entire array value.
    return StoreResult(update, node_map_.at(update->update_value()));
  }

  llvm::Value* original_array = node_map_.at(update->array_to_update());
  llvm::Type* array_type = original_array->getType();
  llvm::AllocaInst* alloca = builder_->CreateAlloca(array_type);
  builder_->CreateStore(original_array, alloca);

  Type* element_type = update->array_to_update()->GetType();
  std::vector<llvm::Value*> gep_indices = {
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx_), 0)};
  llvm::Value* is_inbounds = builder_->getTrue();
  for (Node* index_operand : update->indices()) {
    llvm::Value* index = node_map_.at(index_operand);

    int64_t index_bitwidth = index->getType()->getIntegerBitWidth();
    int64_t comparison_bitwidth = std::max(index_bitwidth, int64_t{64});
    llvm::Value* array_size_comparison_bitwidth =
        llvm::ConstantInt::get(llvm::Type::getIntNTy(ctx_, comparison_bitwidth),
                               element_type->AsArrayOrDie()->size());
    llvm::Value* index_value_comparison_bitwidth = builder_->CreateZExt(
        index, llvm::Type::getIntNTy(ctx_, comparison_bitwidth));
    llvm::Value* is_index_inbounds = builder_->CreateICmpULT(
        index_value_comparison_bitwidth, array_size_comparison_bitwidth,
        "idx_is_inbounds");

    gep_indices.push_back(index_value_comparison_bitwidth);
    is_inbounds =
        builder_->CreateAnd(is_inbounds, is_index_inbounds, "inbounds");

    element_type = element_type->AsArrayOrDie()->element_type();
  }

  // Create the join block which occurs after the conditional block (conditioned
  // on whether the index is inbounds).
  llvm::BasicBlock* join_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(update->GetName(), "_join"), llvm_fn());

  // Create the inbounds block and fill with a store to the array elemnt.
  llvm::BasicBlock* inbounds_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(update->GetName(), "_inbounds"), llvm_fn(),
      /*InsertBefore=*/join_block);
  llvm::IRBuilder<> inbounds_builder(inbounds_block);
  llvm::Value* gep =
      inbounds_builder.CreateGEP(array_type, alloca, gep_indices);
  inbounds_builder.CreateStore(node_map_.at(update->update_value()), gep);
  inbounds_builder.CreateBr(join_block);

  // Create a conditional branch using the original builder (end of the BB
  // before the if/then).
  builder()->CreateCondBr(is_inbounds, inbounds_block, join_block);

  // Create a new BB at the join point.
  auto join_builder = std::make_unique<llvm::IRBuilder<>>(join_block);
  set_builder(std::move(join_builder));

  llvm::Value* update_array = builder_->CreateLoad(array_type, alloca);
  array_storage_[update_array] = alloca;
  return StoreResult(update, update_array);
}

absl::Status FunctionBuilderVisitor::HandleArrayConcat(ArrayConcat* concat) {
  llvm::Type* array_type =
      type_converter_->ConvertToLlvmType(concat->GetType());

  llvm::Value* result = CreateTypedZeroValue(array_type);

  int64_t result_index = 0;
  int64_t result_elements = array_type->getArrayNumElements();
  for (Node* operand : concat->operands()) {
    llvm::Value* array = node_map_.at(operand);
    llvm::Type* array_type = array->getType();

    int64_t element_count = array_type->getArrayNumElements();
    for (int64_t j = 0; j < element_count; ++j) {
      llvm::Value* element =
          builder_->CreateExtractValue(array, {static_cast<uint32_t>(j)});

      if (result_index >= result_elements) {
        return absl::InternalError(absl::StrFormat(
            "array-concat %s result and source have mismatched number of "
            "elements - expected %d",
            concat->ToString(), result_elements));
      }

      result = builder_->CreateInsertValue(
          result, element, {static_cast<uint32_t>(result_index)});
      ++result_index;
    }
  }

  return StoreResult(concat, result);
}

absl::Status FunctionBuilderVisitor::HandleBitSlice(BitSlice* bit_slice) {
  llvm::Value* value = node_map_.at(bit_slice->operand(0));
  Value shift_amount(
      UBits(bit_slice->start(), value->getType()->getIntegerBitWidth()));
  XLS_ASSIGN_OR_RETURN(
      llvm::Constant * start,
      type_converter_->ToLlvmConstant(value->getType(), shift_amount));

  // Then shift and "mask" (by casting) the input value.
  llvm::Value* shifted_value = builder_->CreateLShr(value, start);
  llvm::Value* truncated_value = builder_->CreateTrunc(
      shifted_value, llvm::IntegerType::get(ctx_, bit_slice->width()));
  return StoreResult(bit_slice, truncated_value);
}

absl::Status FunctionBuilderVisitor::HandleBitSliceUpdate(
    BitSliceUpdate* update) {
  llvm::Value* to_update = node_map_.at(update->to_update());
  llvm::Value* start = node_map_.at(update->start());
  llvm::Value* update_value = node_map_.at(update->update_value());

  llvm::IntegerType* result_type = builder_->getIntNTy(update->BitCountOrDie());

  // Zero extend each value to the max of any of the values' widths.
  int max_width = std::max(std::max(to_update->getType()->getIntegerBitWidth(),
                                    start->getType()->getIntegerBitWidth()),
                           update_value->getType()->getIntegerBitWidth());
  llvm::IntegerType* max_width_type = builder_->getIntNTy(max_width);
  llvm::Value* to_update_wide =
      builder_->CreateZExt(to_update, max_width_type, "to_update");
  llvm::Value* start_wide =
      builder_->CreateZExt(start, max_width_type, "start");
  llvm::Value* update_value_wide =
      builder_->CreateZExt(update_value, max_width_type, "update_value");

  // If start is greater than or equal to the width of to_update, then the
  // updated slice is entirely out of bounds the result of the operation is
  // simply to_update.
  llvm::Value* in_bounds = builder_->CreateICmpULT(
      start_wide,
      llvm::ConstantInt::get(max_width_type,
                             to_update->getType()->getIntegerBitWidth()),
      "start_is_inbounds");

  // Create a mask 00..0011..11 where the number of ones is equal to the
  // width of the update value. Then the updated value is:
  //
  //   (~(mask << start) & to_update) | (update_value << start)
  //
  // The shift is guaranteed to be non-poison because of the start value is
  // guarded by a compare against the width lhs of the shift (max_width).
  XLS_ASSIGN_OR_RETURN(
      llvm::Value * mask,
      type_converter_->ToLlvmConstant(
          update->package()->GetBitsType(max_width),
          Value(bits_ops::ZeroExtend(
              Bits::AllOnes(update->update_value()->BitCountOrDie()),
              max_width))));
  // Set the shift amount to the 0 in the case of overshift (start >= max_width)
  // to avoid creating a poisonous shift value which can run afoul of LLVM
  // optimization bugs. The shifted value is not even used in this case.
  llvm::Value* shift_amount = builder_->CreateSelect(
      in_bounds, start_wide, llvm::ConstantInt::get(max_width_type, 0));
  llvm::Value* shifted_mask =
      builder_->CreateNot(builder_->CreateShl(mask, shift_amount), "mask");
  llvm::Value* masked_to_update =
      builder_->CreateAnd(shifted_mask, to_update_wide);
  llvm::Value* shifted_update_value =
      builder_->CreateShl(update_value_wide, shift_amount);
  llvm::Value* updated_slice = builder_->CreateTrunc(
      builder_->CreateOr(masked_to_update, shifted_update_value), result_type,
      "updated_slice");

  llvm::Value* result =
      builder_->CreateSelect(in_bounds, updated_slice, to_update, "result");

  return StoreResult(update, result);
}

absl::Status FunctionBuilderVisitor::HandleDynamicBitSlice(
    DynamicBitSlice* dynamic_bit_slice) {
  llvm::Value* value = node_map_.at(dynamic_bit_slice->operand(0));
  llvm::Value* start = node_map_.at(dynamic_bit_slice->operand(1));
  int64_t value_width = value->getType()->getIntegerBitWidth();
  int64_t start_width = start->getType()->getIntegerBitWidth();
  // Either value or start may be wider, so we use the widest of both
  // since LLVM requires both arguments to be of the same type for
  // comparison and shifting.
  int64_t max_width = std::max(start_width, value_width);
  llvm::IntegerType* max_width_type = builder_->getIntNTy(max_width);
  llvm::Value* value_ext = builder_->CreateZExt(value, max_width_type);
  llvm::Value* start_ext = builder_->CreateZExt(start, max_width_type);

  Value operand_width(UBits(value_width, max_width));
  XLS_ASSIGN_OR_RETURN(
      llvm::Constant * bit_width,
      type_converter_->ToLlvmConstant(max_width_type, operand_width));

  // "out_of_bounds" indicates whether slice is completely out of bounds.
  llvm::Value* out_of_bounds = builder_->CreateICmpUGE(start_ext, bit_width);
  llvm::IntegerType* return_type =
      llvm::IntegerType::get(ctx_, dynamic_bit_slice->width());
  XLS_ASSIGN_OR_RETURN(
      llvm::Constant * zeros,
      type_converter_->ToLlvmConstant(return_type,
                                      Value(Bits(dynamic_bit_slice->width()))));
  // Then shift and truncate the input value. Set the shift amount to the 0 in
  // the case of overshift to avoid creating a poisonous shift value which can
  // run afoul of LLVM optimization bugs. The shifted value is not even used in
  // this case.
  llvm::Value* shift_amount = builder_->CreateSelect(
      out_of_bounds, llvm::ConstantInt::get(max_width_type, 0), start_ext);
  llvm::Value* shifted_value = builder_->CreateLShr(value_ext, shift_amount);
  llvm::Value* truncated_value =
      builder_->CreateTrunc(shifted_value, return_type);
  llvm::Value* result =
      builder_->CreateSelect(out_of_bounds, zeros, truncated_value);
  return StoreResult(dynamic_bit_slice, result);
}

absl::Status FunctionBuilderVisitor::HandleConcat(Concat* concat) {
  llvm::Type* dest_type = type_converter_->ConvertToLlvmType(concat->GetType());
  llvm::Value* base = llvm::ConstantInt::get(dest_type, 0);

  int current_shift = dest_type->getIntegerBitWidth();
  for (const Node* xls_operand : concat->operands()) {
    // Widen each operand to the full size, shift to the right location, and
    // bitwise or into the result value.
    int64_t operand_width = xls_operand->BitCountOrDie();
    llvm::Value* operand = node_map_.at(xls_operand);
    operand = builder_->CreateZExt(operand, dest_type);
    llvm::Value* shifted_operand =
        builder_->CreateShl(operand, current_shift - operand_width);
    base = builder_->CreateOr(base, shifted_operand);

    current_shift -= operand_width;
  }

  return StoreResult(concat, base);
}

absl::Status FunctionBuilderVisitor::HandleCountedFor(CountedFor* counted_for) {
  XLS_ASSIGN_OR_RETURN(llvm::Function * function,
                       GetModuleFunction(counted_for->body()));
  // Add the loop carry and the index to the invariant arguments.
  std::vector<llvm::Value*> args(counted_for->invariant_args().size() + 2);
  for (int i = 0; i < counted_for->invariant_args().size(); i++) {
    args[i + 2] = node_map_.at(counted_for->invariant_args()[i]);
  }

  args[1] = node_map_.at(counted_for->initial_value());

  auto required = GetRequiredArgs();
  args.insert(args.end(), required.begin(), required.end());

  llvm::Type* function_type = function->getType()->getPointerElementType();
  for (int i = 0; i < counted_for->trip_count(); ++i) {
    args[0] = llvm::ConstantInt::get(function_type->getFunctionParamType(0),
                                     i * counted_for->stride());
    args[1] = builder_->CreateCall(function, args);
  }

  return StoreResult(counted_for, args[1]);
}

absl::Status FunctionBuilderVisitor::HandleDecode(Decode* decode) {
  llvm::Value* input = node_map_.at(decode->operand(0));
  llvm::Type* result_type = llvm::IntegerType::get(ctx_, decode->width());
  // If the input value is greater than this op's width, then return 0.
  // In that case, the shl will produce a poison value, but it'll be unused.
  llvm::Value* cast_input = builder_->CreateZExt(input, result_type);
  llvm::Value* overflow = builder_->CreateICmpUGE(
      cast_input, llvm::ConstantInt::get(result_type, decode->width()));
  llvm::Value* result = builder_->CreateSelect(
      overflow, llvm::ConstantInt::get(result_type, 0),
      builder_->CreateShl(llvm::ConstantInt::get(result_type, 1), cast_input));

  return StoreResult(decode, result);
}

absl::Status FunctionBuilderVisitor::HandleDynamicCountedFor(
    DynamicCountedFor* dynamic_counted_for) {
  // Grab loop body.
  XLS_ASSIGN_OR_RETURN(llvm::Function * loop_body_function,
                       GetModuleFunction(dynamic_counted_for->body()));
  llvm::Type* loop_body_function_type =
      loop_body_function->getType()->getPointerElementType();

  // The loop body arguments are the invariant arguments plus the loop carry and
  // index.
  std::vector<llvm::Value*> args(dynamic_counted_for->invariant_args().size() +
                                 2);
  for (int i = 0; i < dynamic_counted_for->invariant_args().size(); i++) {
    args[i + 2] = node_map_.at(dynamic_counted_for->invariant_args()[i]);
  }

  auto required = GetRequiredArgs();
  args.insert(args.end(), required.begin(), required.end());

  // Create basic blocks and corresponding builders. We have 4 blocks:
  // Entry     - the code executed before the loop.
  // Preheader - checks the loop condition.
  // Loop      - calls the loop body, updates the loop carry and index.
  // Exit      - the block we jump to after the loop.
  // ------------------------------------------------------------------
  // Entry
  llvm::Function* llvm_function = builder_->GetInsertBlock()->getParent();
  llvm::BasicBlock* entry_block = builder_->GetInsertBlock();
  llvm::IRBuilder<>* entry_builder = builder_.get();
  // Preheader
  llvm::BasicBlock* preheader_block = llvm::BasicBlock::Create(
      ctx_, absl::StrCat(dynamic_counted_for->GetName(), "_preheader"),
      llvm_function);
  std::unique_ptr<llvm::IRBuilder<>> preheader_builder =
      std::make_unique<llvm::IRBuilder<>>(preheader_block);
  // Loop
  llvm::BasicBlock* loop_block = llvm::BasicBlock::Create(
      ctx_, absl::StrCat(dynamic_counted_for->GetName(), "_loop_block"),
      llvm_function);
  std::unique_ptr<llvm::IRBuilder<>> loop_builder =
      std::make_unique<llvm::IRBuilder<>>(loop_block);
  // Exit
  llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(
      ctx_, absl::StrCat(dynamic_counted_for->GetName(), "_exit_block"),
      llvm_function);
  std::unique_ptr<llvm::IRBuilder<>> exit_builder =
      std::make_unique<llvm::IRBuilder<>>(exit_block);

  // Get initial index, loop carry.
  llvm::Type* index_type = loop_body_function_type->getFunctionParamType(0);
  llvm::Value* init_index = llvm::ConstantInt::get(index_type, 0);
  llvm::Value* init_loop_carry =
      node_map_.at(dynamic_counted_for->initial_value());

  // In entry, grab trip_count and stride, extended to match index type.
  // trip_count is zero-extended because the input trip_count is treated as
  // to be unsigned while stride is treated as signed.
  llvm::Value* trip_count = entry_builder->CreateZExt(
      node_map_.at(dynamic_counted_for->trip_count()), index_type);
  llvm::Value* stride = entry_builder->CreateSExt(
      node_map_.at(dynamic_counted_for->stride()), index_type);

  // Calculate index limit and jump entry loop predheader.
  llvm::Value* index_limit = entry_builder->CreateMul(trip_count, stride);
  entry_builder->CreateBr(preheader_block);

  // Preheader
  // Check if trip_count interations completed.
  // If so, exit loop. Otherwise, keep looping.
  llvm::PHINode* index_phi = preheader_builder->CreatePHI(index_type, 2);
  args[0] = index_phi;
  llvm::PHINode* loop_carry_phi =
      preheader_builder->CreatePHI(init_loop_carry->getType(), 2);
  args[1] = loop_carry_phi;
  llvm::Value* index_limit_reached =
      preheader_builder->CreateICmpEQ(index_phi, index_limit);
  preheader_builder->CreateCondBr(index_limit_reached, exit_block, loop_block);

  // Loop
  // Call loop body function and increment index before returning to
  // preheader_builder.
  llvm::Value* loop_carry =
      loop_builder->CreateCall(loop_body_function, {args});
  llvm::Value* inc_index = loop_builder->CreateAdd(index_phi, stride);
  loop_builder->CreateBr(preheader_block);

  // Set predheader Phi node inputs.
  index_phi->addIncoming(init_index, entry_block);
  index_phi->addIncoming(inc_index, loop_block);
  loop_carry_phi->addIncoming(init_loop_carry, entry_block);
  loop_carry_phi->addIncoming(loop_carry, loop_block);

  // Add a single-input PHI node for loop carry output so that,
  // under single static assignment, the loop carry is only used
  // in the loop. Llvm should do this for us in llvm::formLCSSA when we JIT
  // compile, but for some reason that functions fails it's own assertion
  // (L.isLCSSAForm(DT)) after its transformation is applied. So, we do this
  // manually here.
  llvm::PHINode* loop_carry_out =
      exit_builder->CreatePHI(init_loop_carry->getType(), 1);
  loop_carry_out->addIncoming(loop_carry_phi, preheader_block);

  // Set the builder to build in the exit block.
  set_builder(std::move(exit_builder));

  return StoreResult(dynamic_counted_for, loop_carry_out);
}

absl::Status FunctionBuilderVisitor::HandleEncode(Encode* encode) {
  llvm::Value* input = node_map_.at(encode->operand(0));
  llvm::Type* input_type = input->getType();
  llvm::Value* input_one = llvm::ConstantInt::get(input_type, 1);

  llvm::Type* result_type =
      type_converter_->ConvertToLlvmType(encode->GetType());
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

absl::StatusOr<std::vector<FunctionBuilderVisitor::CompareTerm>>
FunctionBuilderVisitor::ExpandTerms(Node* lhs, Node* rhs, Node* src) {
  XLS_RET_CHECK(lhs->GetType() == rhs->GetType()) << absl::StreamFormat(
      "The lhs and rhs of %s have different types: lhs %s rhs %s",
      src->ToString(), lhs->GetType()->ToString(), rhs->GetType()->ToString());

  struct ToExpand {
    Type* ty;
    llvm::Value* lhs;
    llvm::Value* rhs;
  };

  std::vector<ToExpand> unexpanded = {
      ToExpand{lhs->GetType(), node_map_.at(lhs), node_map_.at(rhs)}};

  std::vector<CompareTerm> terms;

  while (!unexpanded.empty()) {
    ToExpand next = unexpanded.back();
    unexpanded.pop_back();

    switch (next.ty->kind()) {
      case TypeKind::kToken:
        // Tokens represent different points in time and are incomparable.
        return absl::InvalidArgumentError(absl::StrFormat(
            "Tokens are incomparable so this expression is illegal: %s",
            src->ToString()));
      case TypeKind::kBits:
        terms.push_back(CompareTerm{next.lhs, next.rhs});
        break;
      case TypeKind::kArray: {
        ArrayType* array_type = next.ty->AsArrayOrDie();
        Type* element_type = array_type->element_type();
        // Cast once so we do not have to cast when calling CreateExtractValue
        uint32_t array_size = static_cast<uint32_t>(array_type->size());
        for (uint32_t i = 0; i < array_size; i++) {
          llvm::Value* lhs_value = builder_->CreateExtractValue(next.lhs, {i});
          llvm::Value* rhs_value = builder_->CreateExtractValue(next.rhs, {i});
          unexpanded.push_back(ToExpand{element_type, lhs_value, rhs_value});
        }
        break;
      }
      case TypeKind::kTuple: {
        TupleType* tuple_type = next.ty->AsTupleOrDie();
        // Cast once so we do not have to cast when calling CreateExtractValue
        uint32_t tuple_size = static_cast<uint32_t>(tuple_type->size());
        for (uint32_t i = 0; i < tuple_size; i++) {
          Type* element_type = tuple_type->element_type(i);
          llvm::Value* lhs_value = builder_->CreateExtractValue(next.lhs, {i});
          llvm::Value* rhs_value = builder_->CreateExtractValue(next.rhs, {i});
          unexpanded.push_back(ToExpand{element_type, lhs_value, rhs_value});
        }
        break;
      }
    }
  }
  return terms;
}

absl::Status FunctionBuilderVisitor::HandleEq(CompareOp* eq) {
  Node* lhs = eq->operand(0);
  Node* rhs = eq->operand(1);

  XLS_ASSIGN_OR_RETURN(std::vector<CompareTerm> eq_terms,
                       ExpandTerms(lhs, rhs, eq));

  llvm::Value* result = builder_->getTrue();

  for (const auto& eq_term : eq_terms) {
    llvm::Value* term_test = builder_->CreateICmpEQ(eq_term.lhs, eq_term.rhs);
    result = builder_->CreateAnd(result, term_test);
  }

  return StoreResult(eq, result);
}

absl::Status FunctionBuilderVisitor::HandleIdentity(UnOp* identity) {
  return StoreResult(identity, node_map_.at(identity->operand(0)));
}

absl::Status FunctionBuilderVisitor::HandleInvoke(Invoke* invoke) {
  XLS_ASSIGN_OR_RETURN(llvm::Function * function,
                       GetModuleFunction(invoke->to_apply()));

  std::vector<llvm::Value*> args(invoke->operand_count());
  for (int i = 0; i < invoke->operand_count(); i++) {
    args[i] = node_map_[invoke->operand(i)];
  }

  auto required = GetRequiredArgs();
  args.insert(args.end(), required.begin(), required.end());

  llvm::Value* invoke_inst = builder_->CreateCall(function, args);
  return StoreResult(invoke, invoke_inst);
}

absl::Status FunctionBuilderVisitor::HandleLiteral(Literal* literal) {
  Type* xls_type = literal->GetType();
  XLS_ASSIGN_OR_RETURN(
      llvm::Value * llvm_literal,
      type_converter_->ToLlvmConstant(xls_type, literal->value()));

  return StoreResult(literal, llvm_literal);
}

absl::Status FunctionBuilderVisitor::HandleMap(Map* map) {
  XLS_ASSIGN_OR_RETURN(llvm::Function * to_apply,
                       GetModuleFunction(map->to_apply()));

  llvm::Value* input = node_map_.at(map->operand(0));
  llvm::Type* input_type = input->getType();
  llvm::FunctionType* function_type = llvm::cast<llvm::FunctionType>(
      to_apply->getType()->getPointerElementType());

  llvm::Value* result = CreateTypedZeroValue(llvm::ArrayType::get(
      function_type->getReturnType(), input_type->getArrayNumElements()));

  for (uint32_t i = 0; i < input_type->getArrayNumElements(); ++i) {
    llvm::Value* map_arg = builder_->CreateExtractValue(input, {i});

    std::vector<llvm::Value*> iter_args = GetRequiredArgs();
    // The argument of the map function goes before the required arguments.
    iter_args.insert(iter_args.begin(), map_arg);

    llvm::Value* iter_result = builder_->CreateCall(to_apply, iter_args);
    result = builder_->CreateInsertValue(result, iter_result, {i});
  }

  return StoreResult(map, result);
}

absl::Status FunctionBuilderVisitor::HandleSMul(ArithOp* mul) {
  return HandleArithOp(mul);
}

absl::Status FunctionBuilderVisitor::HandleUMul(ArithOp* mul) {
  return HandleArithOp(mul);
}

absl::Status FunctionBuilderVisitor::HandleNaryAnd(NaryOp* and_op) {
  llvm::Value* result = node_map_.at((and_op->operand(0)));
  for (int i = 1; i < and_op->operand_count(); ++i) {
    result = builder_->CreateAnd(result, node_map_.at(and_op->operand(i)));
  }
  return StoreResult(and_op, result);
}

absl::Status FunctionBuilderVisitor::HandleNaryNand(NaryOp* nand_op) {
  llvm::Value* result = node_map_.at((nand_op->operand(0)));
  for (int i = 1; i < nand_op->operand_count(); ++i) {
    result = builder_->CreateAnd(result, node_map_.at(nand_op->operand(i)));
  }
  result = builder_->CreateNot(result);
  return StoreResult(nand_op, result);
}

absl::Status FunctionBuilderVisitor::HandleNaryNor(NaryOp* nor_op) {
  llvm::Value* result = node_map_.at((nor_op->operand(0)));
  for (int i = 1; i < nor_op->operand_count(); ++i) {
    result = builder_->CreateOr(result, node_map_.at(nor_op->operand(i)));
  }
  result = builder_->CreateNot(result);
  return StoreResult(nor_op, result);
}

absl::Status FunctionBuilderVisitor::HandleNaryOr(NaryOp* or_op) {
  llvm::Value* result = node_map_.at((or_op->operand(0)));
  for (int i = 1; i < or_op->operand_count(); ++i) {
    result = builder_->CreateOr(result, node_map_.at(or_op->operand(i)));
  }
  return StoreResult(or_op, result);
}

absl::Status FunctionBuilderVisitor::HandleNaryXor(NaryOp* xor_op) {
  llvm::Value* result = node_map_.at((xor_op->operand(0)));
  for (int i = 1; i < xor_op->operand_count(); ++i) {
    result = builder_->CreateXor(result, node_map_.at(xor_op->operand(i)));
  }
  return StoreResult(xor_op, result);
}

absl::Status FunctionBuilderVisitor::HandleNe(CompareOp* ne) {
  Node* lhs = ne->operand(0);
  Node* rhs = ne->operand(1);

  XLS_ASSIGN_OR_RETURN(std::vector<CompareTerm> ne_terms,
                       ExpandTerms(lhs, rhs, ne));

  llvm::Value* result = builder_->getFalse();

  for (const auto& ne_term : ne_terms) {
    llvm::Value* term_test = builder_->CreateICmpNE(ne_term.lhs, ne_term.rhs);
    result = builder_->CreateOr(result, term_test);
  }

  return StoreResult(ne, result);
}

absl::Status FunctionBuilderVisitor::HandleNeg(UnOp* neg) {
  llvm::Value* llvm_neg = builder_->CreateNeg(node_map_.at(neg->operand(0)));
  return StoreResult(neg, llvm_neg);
}

absl::Status FunctionBuilderVisitor::HandleNot(UnOp* not_op) {
  llvm::Value* llvm_not = builder_->CreateNot(node_map_.at(not_op->operand(0)));
  return StoreResult(not_op, llvm_not);
}

absl::Status FunctionBuilderVisitor::HandleOneHot(OneHot* one_hot) {
  llvm::Value* input = node_map_.at(one_hot->operand(0));
  llvm::Type* input_type = input->getType();
  int input_width = input_type->getIntegerBitWidth();

  llvm::Value* llvm_false = builder_->getFalse();
  llvm::Value* llvm_true = builder_->getTrue();

  // Special case the 0-bit input value, it produces a single true output bit.
  if (one_hot->operand(0)->GetType()->AsBitsOrDie()->bit_count() == 0) {
    return StoreResult(one_hot, llvm_true);
  }

  llvm::Value* zeroes;
  if (one_hot->priority() == LsbOrMsb::kLsb) {
    llvm::Function* cttz = llvm::Intrinsic::getDeclaration(
        module_, llvm::Intrinsic::cttz, {input_type});
    // We don't need to pass user data to these intrinsics; they're leaf nodes.
    zeroes = builder_->CreateCall(cttz, {input, llvm_false});
  } else {
    llvm::Function* ctlz = llvm::Intrinsic::getDeclaration(
        module_, llvm::Intrinsic::ctlz, {input_type});
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

absl::Status FunctionBuilderVisitor::HandleOneHotSel(OneHotSelect* sel) {
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

absl::Status FunctionBuilderVisitor::HandleOrReduce(BitwiseReductionOp* op) {
  // OR-reduce is equivalent to checking if any bit is set in the input.
  llvm::Value* operand = node_map_.at(op->operand(0));
  llvm::Value* eq = builder_->CreateICmpNE(
      operand, llvm::ConstantInt::get(operand->getType(), 0));
  return StoreResult(op, eq);
}

absl::Status FunctionBuilderVisitor::HandleParam(Param* param) {
  // If we're not processing the first function in LLVM space, this is easy -
  // just return the n'th argument to the active function.
  //
  // If this IS that entry function, then we need to pull in data from the
  // opaque arg buffer:
  //  1. Find out the index of the param we're loading.
  //  2. Get the offset of that param into our arg buffer.
  //  3. Cast that offset/pointer into the target type and load from it.
  XLS_ASSIGN_OR_RETURN(int index, param->function_base()->GetParamIndex(param));
  llvm::Function* llvm_function = builder_->GetInsertBlock()->getParent();

  if (!is_top_) {
    return StoreResult(param, llvm_function->getArg(index));
  }

  if (param->GetType()->IsToken()) {
    return StoreResult(param, type_converter_->GetToken());
  }

  // Just handle tokens here, since they're "nothing".
  if (generate_packed_ && !param->GetType()->IsToken()) {
    return HandlePackedParam(param);
  }

  // Remember that all input arg pointers are packed into a buffer specified
  // as a single formal parameter, hence the 0 constant here.
  llvm::Argument* arg_pointer = llvm_function->getArg(0);

  llvm::Type* arg_type = type_converter_->ConvertToLlvmType(param->GetType());
  llvm::Type* llvm_arg_ptr_type =
      llvm::PointerType::get(arg_type, /*AddressSpace=*/0);

  // Load 1: Get the pointer to arg N out of memory (the arg redirect buffer).
  llvm::Value* gep = builder_->CreateGEP(
      arg_pointer,
      {
          llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx_), 0),
          llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx_), index),
      });
  llvm::LoadInst* load =
      builder_->CreateLoad(gep->getType()->getPointerElementType(), gep);
  llvm::Value* cast = builder_->CreateBitCast(load, llvm_arg_ptr_type);

  // Load 2: Get the data at that pointer's destination.
  load = builder_->CreateLoad(arg_type, cast);

  return StoreResult(param, load);
}

absl::Status FunctionBuilderVisitor::HandlePackedParam(Param* param) {
  // For packed params, we need to get the buffer holding the param of
  // interest, then decompose it into the structure expected by LLVM.
  // That structure is target-dependent, but in general, has each tuple or
  // array element aligned to a byte (or larger) boundary (e.g., as storing
  // an i24 in an i32).
  llvm::Function* llvm_function = builder_->GetInsertBlock()->getParent();
  llvm::Argument* arg_pointer = llvm_function->getArg(0);
  XLS_ASSIGN_OR_RETURN(int index, param->function_base()->GetParamIndex(param));

  // First, load the arg buffer (as an i8*).
  // Then pull out elements from that buffer to make the final type.
  // Load 1: Get the pointer to arg N out of memory (the arg redirect buffer).
  llvm::Value* gep = builder_->CreateGEP(
      arg_pointer,
      {
          llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx_), 0),
          llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx_), index),
      });

  // The GEP gives a pointer to a u8*; so 'load' is an i8*. Cast it to its full
  // width so we can load the whole thing.
  llvm::Type* int8_ptr_type = llvm::Type::getInt8PtrTy(ctx(), 0);
  llvm::LoadInst* load = builder_->CreateLoad(int8_ptr_type, gep);
  if (param->GetType()->GetFlatBitCount() == 0) {
    // Create an empty structure, etc.
    return StoreResult(
        param, CreateTypedZeroValue(
                   type_converter_->ConvertToLlvmType(param->GetType())));
  }
  llvm::Type* packed_arg_type =
      llvm::IntegerType::get(ctx_, param->GetType()->GetFlatBitCount());
  llvm::Value* cast = builder_->CreateBitCast(
      load, llvm::PointerType::get(packed_arg_type, /*AddressSpace=*/0));
  load = builder_->CreateLoad(packed_arg_type, cast);

  // Now populate an Value of Param's type with the packed buffer contents.
  XLS_ASSIGN_OR_RETURN(llvm::Value * unpacked,
                       UnpackParamBuffer(param->GetType(), load));

  return StoreResult(param, unpacked);
}

// param_buffer is an LLVM i8 (not a pointer to such). So for each element in
// the param [type], we read it from the buffer, then shift off the read
// amount.
absl::StatusOr<llvm::Value*> FunctionBuilderVisitor::UnpackParamBuffer(
    Type* param_type, llvm::Value* param_buffer) {
  switch (param_type->kind()) {
    case TypeKind::kBits:
      return builder_->CreateTrunc(
          param_buffer,
          llvm::IntegerType::get(ctx_, param_type->GetFlatBitCount()));
    case TypeKind::kArray: {
      // Create an empty array and plop in every element.
      ArrayType* array_type = param_type->AsArrayOrDie();
      Type* element_type = array_type->element_type();

      llvm::Value* array =
          CreateTypedZeroValue(type_converter_->ConvertToLlvmType(array_type));
      for (uint32_t i = 0; i < array_type->size(); i++) {
        XLS_ASSIGN_OR_RETURN(llvm::Value * element,
                             UnpackParamBuffer(element_type, param_buffer));
        array = builder_->CreateInsertValue(array, element, {i});
        param_buffer =
            builder_->CreateLShr(param_buffer, element_type->GetFlatBitCount());
      }
      return array;
    }
    case TypeKind::kTuple: {
      // Create an empty tuple and plop in every element.
      TupleType* tuple_type = param_type->AsTupleOrDie();
      llvm::Value* tuple =
          CreateTypedZeroValue(type_converter_->ConvertToLlvmType(tuple_type));
      for (int32_t i = tuple_type->size() - 1; i >= 0; i--) {
        // Tuple elements are stored MSB -> LSB, so we need to extract in
        // reverse order to match native layout.
        Type* element_type = tuple_type->element_type(i);
        XLS_ASSIGN_OR_RETURN(llvm::Value * element,
                             UnpackParamBuffer(element_type, param_buffer));
        tuple = builder_->CreateInsertValue(tuple, element,
                                            {static_cast<uint32_t>(i)});
        param_buffer =
            builder_->CreateLShr(param_buffer, element_type->GetFlatBitCount());
      }
      return tuple;
    }
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unhandled type kind: ", TypeKindToString(param_type->kind())));
  }
}

absl::Status FunctionBuilderVisitor::HandleReverse(UnOp* reverse) {
  llvm::Value* input = node_map_.at(reverse->operand(0));
  llvm::Function* reverse_fn = llvm::Intrinsic::getDeclaration(
      module_, llvm::Intrinsic::bitreverse, {input->getType()});
  // Intrinsics don't need user_data ptrs; they're leaf nodes.
  return StoreResult(reverse, builder_->CreateCall(reverse_fn, {input}));
}

absl::Status FunctionBuilderVisitor::HandleSDiv(BinOp* binop) {
  return HandleBinOp(binop);
}

absl::Status FunctionBuilderVisitor::HandleSMod(BinOp* binop) {
  return HandleBinOp(binop);
}

absl::Status FunctionBuilderVisitor::HandleSel(Select* sel) {
  // Sel is implemented by a cascading series of select ops, e.g.,
  // selector == 0 ? cases[0] : selector == 1 ? cases[1] : selector == 2 ? ...
  llvm::Value* selector = node_map_.at(sel->selector());
  llvm::Value* llvm_sel =
      sel->default_value() ? node_map_.at(*sel->default_value()) : nullptr;
  for (int i = sel->cases().size() - 1; i >= 0; i--) {
    Node* node = sel->get_case(i);
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

absl::Status FunctionBuilderVisitor::HandleSGe(CompareOp* ge) {
  llvm::Value* lhs = node_map_.at(ge->operand(0));
  llvm::Value* rhs = node_map_.at(ge->operand(1));
  llvm::Value* result = builder_->CreateICmpSGE(lhs, rhs);
  return StoreResult(ge, result);
}

absl::Status FunctionBuilderVisitor::HandleSGt(CompareOp* gt) {
  llvm::Value* lhs = node_map_.at(gt->operand(0));
  llvm::Value* rhs = node_map_.at(gt->operand(1));
  llvm::Value* result = builder_->CreateICmpSGT(lhs, rhs);
  return StoreResult(gt, result);
}

absl::Status FunctionBuilderVisitor::HandleSignExtend(ExtendOp* sign_ext) {
  llvm::Type* new_type =
      llvm::IntegerType::get(ctx_, sign_ext->new_bit_count());
  return StoreResult(
      sign_ext,
      builder_->CreateSExt(node_map_.at(sign_ext->operand(0)), new_type));
}

absl::Status FunctionBuilderVisitor::HandleSLe(CompareOp* le) {
  llvm::Value* lhs = node_map_.at(le->operand(0));
  llvm::Value* rhs = node_map_.at(le->operand(1));
  llvm::Value* result = builder_->CreateICmpSLE(lhs, rhs);
  return StoreResult(le, result);
}

absl::Status FunctionBuilderVisitor::HandleSLt(CompareOp* lt) {
  llvm::Value* lhs = node_map_.at(lt->operand(0));
  llvm::Value* rhs = node_map_.at(lt->operand(1));
  llvm::Value* result = builder_->CreateICmpSLT(lhs, rhs);
  return StoreResult(lt, result);
}

absl::Status FunctionBuilderVisitor::HandleShll(BinOp* binop) {
  return HandleBinOp(binop);
}

absl::Status FunctionBuilderVisitor::HandleShra(BinOp* binop) {
  return HandleBinOp(binop);
}

absl::Status FunctionBuilderVisitor::HandleShrl(BinOp* binop) {
  return HandleBinOp(binop);
}

absl::Status FunctionBuilderVisitor::HandleSub(BinOp* binop) {
  return HandleBinOp(binop);
}

absl::Status FunctionBuilderVisitor::HandleTuple(Tuple* tuple) {
  llvm::Type* tuple_type = type_converter_->ConvertToLlvmType(tuple->GetType());

  llvm::Value* result = CreateTypedZeroValue(tuple_type);
  for (uint32_t i = 0; i < tuple->operand_count(); ++i) {
    if (tuple->operand(i)->GetType()->GetFlatBitCount() == 0) {
      continue;
    }
    result = builder_->CreateInsertValue(result,
                                         node_map_.at(tuple->operand(i)), {i});
  }

  return StoreResult(tuple, result);
}

absl::Status FunctionBuilderVisitor::HandleTupleIndex(TupleIndex* index) {
  llvm::Value* value = builder_->CreateExtractValue(
      node_map_.at(index->operand(0)), index->index());
  return StoreResult(index, value);
}

absl::Status FunctionBuilderVisitor::HandleUDiv(BinOp* binop) {
  return HandleBinOp(binop);
}

absl::Status FunctionBuilderVisitor::HandleUMod(BinOp* binop) {
  return HandleBinOp(binop);
}

absl::Status FunctionBuilderVisitor::HandleUGe(CompareOp* ge) {
  llvm::Value* lhs = node_map_.at(ge->operand(0));
  llvm::Value* rhs = node_map_.at(ge->operand(1));
  llvm::Value* result = builder_->CreateICmpUGE(lhs, rhs);
  return StoreResult(ge, result);
}

absl::Status FunctionBuilderVisitor::HandleUGt(CompareOp* gt) {
  llvm::Value* lhs = node_map_.at(gt->operand(0));
  llvm::Value* rhs = node_map_.at(gt->operand(1));
  llvm::Value* result = builder_->CreateICmpUGT(lhs, rhs);
  return StoreResult(gt, result);
}

absl::Status FunctionBuilderVisitor::HandleULe(CompareOp* le) {
  llvm::Value* lhs = node_map_.at(le->operand(0));
  llvm::Value* rhs = node_map_.at(le->operand(1));
  llvm::Value* result = builder_->CreateICmpULE(lhs, rhs);
  return StoreResult(le, result);
}

absl::Status FunctionBuilderVisitor::HandleULt(CompareOp* lt) {
  llvm::Value* lhs = node_map_.at(lt->operand(0));
  llvm::Value* rhs = node_map_.at(lt->operand(1));
  llvm::Value* result = builder_->CreateICmpULT(lhs, rhs);
  return StoreResult(lt, result);
}

absl::Status FunctionBuilderVisitor::HandleXorReduce(BitwiseReductionOp* op) {
  // XOR-reduce is equivalent to checking if the number of set bits is odd.
  llvm::Value* operand = node_map_.at(op->operand(0));
  llvm::Function* ctpop = llvm::Intrinsic::getDeclaration(
      module_, llvm::Intrinsic::ctpop, {operand->getType()});
  // We don't need to pass user data to intrinsics; they're leaf nodes.
  llvm::Value* pop_count = builder_->CreateCall(ctpop, {operand});

  // Once we have the pop count, truncate to the first (i.e., "is odd") bit.
  llvm::Value* truncated_value =
      builder_->CreateTrunc(pop_count, llvm::IntegerType::get(ctx_, 1));
  return StoreResult(op, truncated_value);
}

absl::Status FunctionBuilderVisitor::HandleZeroExtend(ExtendOp* zero_ext) {
  llvm::Value* base = node_map_.at(zero_ext->operand(0));
  llvm::Type* dest_type =
      base->getType()->getWithNewBitWidth(zero_ext->new_bit_count());
  llvm::Value* zext =
      builder_->CreateZExt(node_map_.at(zero_ext->operand(0)), dest_type);
  return StoreResult(zero_ext, zext);
}

absl::Status FunctionBuilderVisitor::HandleArithOp(ArithOp* arith_op) {
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
      type_converter_->ConvertToLlvmType(arith_op->GetType());
  llvm::Value* lhs = builder_->CreateIntCast(node_map_.at(arith_op->operand(0)),
                                             result_type, is_signed);
  llvm::Value* rhs = builder_->CreateIntCast(node_map_.at(arith_op->operand(1)),
                                             result_type, is_signed);

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

absl::Status FunctionBuilderVisitor::HandleBinOp(BinOp* binop) {
  if (binop->operand_count() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected 2 args to a binary op; instead got %d",
                        binop->operand_count()));
  }

  llvm::Value* lhs = node_map_.at(binop->operand(0));
  llvm::Value* rhs = node_map_.at(binop->operand(1));
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
    case Op::kUMod:
      result = EmitMod(lhs, rhs, /*is_signed=*/false);
      break;
    case Op::kSMod:
      result = EmitMod(lhs, rhs, /*is_signed=*/true);
      break;
    default:
      return absl::UnimplementedError(
          absl::StrFormat("Unsupported/unimplemented bin op: %d",
                          static_cast<int>(binop->op())));
  }

  return StoreResult(binop, result);
}

llvm::Value* FunctionBuilderVisitor::EmitShiftOp(Op op, llvm::Value* lhs,
                                                 llvm::Value* rhs) {
  // Shift operands are allowed to be different sizes in the [XLS] IR, so
  // we need to cast them to be the same size here (for LLVM).
  int common_width = std::max(lhs->getType()->getIntegerBitWidth(),
                              rhs->getType()->getIntegerBitWidth());
  llvm::Type* dest_type = llvm::IntegerType::get(ctx_, common_width);
  llvm::Value* wide_lhs = op == Op::kShra
                              ? builder_->CreateSExt(lhs, dest_type)
                              : builder_->CreateZExt(lhs, dest_type);
  llvm::Value* wide_rhs = builder_->CreateZExt(rhs, dest_type);
  // In LLVM, an overshifted shift creates poison.
  llvm::Value* is_overshift = builder_->CreateICmpUGE(
      wide_rhs, llvm::ConstantInt::get(dest_type, common_width));

  llvm::Value* inst;
  llvm::Value* zero = llvm::ConstantInt::get(dest_type, 0);
  llvm::Value* overshift_value = zero;
  // In the event of potential overshift (shift amount >= width of operand)
  // replace the shift amount with zero to avoid creating a poison value. The
  // motiviation is that LLVM has buggy optimizations which sometimes improperly
  // propagate poison values, particularly through selects. In the case where
  // the shift amount is replaced with zero, the shift valued is not even used
  // (selected in the Select instruction) so correctness is not affected.
  llvm::Value* safe_rhs = builder_->CreateSelect(is_overshift, zero, wide_rhs);

  if (op == Op::kShll) {
    inst = builder_->CreateShl(wide_lhs, safe_rhs);
  } else if (op == Op::kShra) {
    llvm::Value* high_bit = builder_->CreateLShr(
        wide_lhs,
        llvm::ConstantInt::get(dest_type,
                               wide_lhs->getType()->getIntegerBitWidth() - 1));
    llvm::Value* high_bit_set =
        builder_->CreateICmpEQ(high_bit, llvm::ConstantInt::get(dest_type, 1));
    overshift_value = builder_->CreateSelect(
        high_bit_set, llvm::ConstantInt::getSigned(dest_type, -1), zero);
    inst = builder_->CreateAShr(wide_lhs, safe_rhs);
  } else {
    inst = builder_->CreateLShr(wide_lhs, safe_rhs);
  }
  llvm::Value* result =
      builder_->CreateSelect(is_overshift, overshift_value, inst);
  // The expected return type is the same as the original lhs. The shifted value
  // may be wider than this, so trucate in that case. This occurs if the
  // original rhs is wider than the original lhs.
  return result->getType() == lhs->getType()
             ? result
             : builder_->CreateTrunc(result, lhs->getType());
}

llvm::Value* FunctionBuilderVisitor::EmitDiv(llvm::Value* lhs, llvm::Value* rhs,
                                             bool is_signed) {
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
  rhs = builder_->CreateSelect(rhs_eq_zero,
                               llvm::ConstantInt::get(rhs->getType(), 1), rhs);
  if (is_signed) {
    llvm::Value* max_value =
        type_converter_
            ->ToLlvmConstant(rhs->getType(), Value(Bits::MaxSigned(type_width)))
            .value();
    llvm::Value* min_value =
        type_converter_
            ->ToLlvmConstant(rhs->getType(), Value(Bits::MinSigned(type_width)))
            .value();

    lhs = builder_->CreateSelect(
        rhs_eq_zero, builder_->CreateSelect(lhs_gt_zero, max_value, min_value),
        lhs);
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

llvm::Value* FunctionBuilderVisitor::EmitMod(llvm::Value* lhs, llvm::Value* rhs,
                                             bool is_signed) {
  // XLS mod semantics differ from LLVMs with regard to mod by zero. In XLS,
  // modulo by zero returns zero rather than undefined behavior.
  llvm::Value* zero = llvm::ConstantInt::get(rhs->getType(), 0);
  llvm::Value* rhs_eq_zero = builder_->CreateICmpEQ(rhs, zero);
  // Replace a zero rhs with one to avoid SIGFPE even though the result is not
  // used.
  rhs = builder_->CreateSelect(rhs_eq_zero,
                               llvm::ConstantInt::get(rhs->getType(), 1), rhs);
  return builder_->CreateSelect(rhs_eq_zero, zero,
                                is_signed ? builder_->CreateSRem(lhs, rhs)
                                          : builder_->CreateURem(lhs, rhs));
}

llvm::Constant* FunctionBuilderVisitor::CreateTypedZeroValue(llvm::Type* type) {
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

llvm::Value* FunctionBuilderVisitor::CreateAggregateOr(llvm::Value* lhs,
                                                       llvm::Value* rhs) {
  llvm::Type* arg_type = lhs->getType();
  if (arg_type->isIntegerTy()) {
    return builder_->CreateOr(lhs, rhs);
  }

  llvm::Value* result = CreateTypedZeroValue(arg_type);
  int num_elements = arg_type->isArrayTy() ? arg_type->getArrayNumElements()
                                           : arg_type->getNumContainedTypes();
  for (uint32_t i = 0; i < num_elements; ++i) {
    llvm::Value* iter_result =
        CreateAggregateOr(builder_->CreateExtractValue(lhs, {i}),
                          builder_->CreateExtractValue(rhs, {i}));
    result = builder_->CreateInsertValue(result, iter_result, {i});
  }

  return result;
}

absl::StatusOr<llvm::Constant*> FunctionBuilderVisitor::ConvertToLlvmConstant(
    Type* type, const Value& value) {
  if (type->IsBits()) {
    return type_converter_->ToLlvmConstant(
        type_converter_->ConvertToLlvmType(type), value);
  } else if (type->IsTuple()) {
    TupleType* tuple_type = type->AsTupleOrDie();
    std::vector<llvm::Constant*> llvm_elements;
    for (int i = 0; i < tuple_type->size(); ++i) {
      XLS_ASSIGN_OR_RETURN(llvm::Constant * llvm_element,
                           type_converter_->ToLlvmConstant(
                               tuple_type->element_type(i), value.element(i)));
      llvm_elements.push_back(llvm_element);
    }

    llvm::Type* llvm_type = type_converter_->ConvertToLlvmType(type);
    return llvm::ConstantStruct::get(llvm::cast<llvm::StructType>(llvm_type),
                                     llvm_elements);
  } else if (type->IsArray()) {
    const ArrayType* array_type = type->AsArrayOrDie();
    std::vector<llvm::Constant*> elements;
    for (const Value& element : value.elements()) {
      XLS_ASSIGN_OR_RETURN(
          llvm::Constant * llvm_element,
          type_converter_->ToLlvmConstant(array_type->element_type(), element));
      elements.push_back(llvm_element);
    }

    llvm::Type* element_type =
        type_converter_->ConvertToLlvmType(array_type->element_type());
    return llvm::ConstantArray::get(
        llvm::ArrayType::get(element_type, array_type->size()), elements);
  }

  XLS_LOG(FATAL) << "Unknown value kind: " << value.kind();
}

absl::StatusOr<llvm::Function*> FunctionBuilderVisitor::GetModuleFunction(
    Function* xls_function) {
  // If we've not processed this function yet, then do so.
  llvm::Function* found_function = module_->getFunction(xls_function->name());
  if (found_function != nullptr) {
    return found_function;
  }

  // There are a couple of differences between this and entry function
  // visitor initialization such that I think it makes slightly more sense
  // to not factor it into a common block, but it's not clear-cut.
  std::vector<llvm::Type*> param_types(xls_function->params().size() + 2);
  for (int i = 0; i < xls_function->params().size(); ++i) {
    param_types[i] =
        type_converter_->ConvertToLlvmType(xls_function->param(i)->GetType());
  }

  // Treat void pointers as int64_t values at the LLVM IR level.
  // Using an actual pointer type triggers LLVM asserts when compiling
  // in debug mode.
  // TODO(amfv): 2021-04-05 Figure out why and fix void pointer handling.
  llvm::Type* void_ptr_type = llvm::Type::getInt64Ty(ctx());

  // Pointer to assertion status temporary
  param_types.at(param_types.size() - 2) = void_ptr_type;

  // We need to add an extra param to every function call to carry our "user
  // data", i.e., callback info.
  param_types.back() = void_ptr_type;

  Type* return_type = GetEffectiveReturnValue(xls_function)->GetType();
  llvm::Type* llvm_return_type =
      type_converter_->ConvertToLlvmType(return_type);

  llvm::FunctionType* function_type = llvm::FunctionType::get(
      llvm_return_type,
      llvm::ArrayRef<llvm::Type*>(param_types.data(), param_types.size()),
      /*isVarArg=*/false);
  llvm::Function* llvm_function = llvm::cast<llvm::Function>(
      module_
          ->getOrInsertFunction(xls_function->qualified_name(), function_type)
          .getCallee());

  // TODO(rspringer): Need to override this for Procs.
  XLS_RETURN_IF_ERROR(FunctionBuilderVisitor::Visit(
      module_, llvm_function, xls_function, type_converter_,
      /*is_top=*/false, /*generate_packed=*/false));

  return llvm_function;
}

absl::Status FunctionBuilderVisitor::StoreResult(Node* node,
                                                 llvm::Value* value) {
  XLS_RET_CHECK(!node_map_.contains(node));
  value->setName(verilog::SanitizeIdentifier(node->GetName()));
  if (node == GetEffectiveReturnValue(node->function_base())) {
    return_value_ = value;
  }
  node_map_[node] = value;

  return absl::OkStatus();
}

absl::StatusOr<llvm::Value*> FunctionBuilderVisitor::PackElement(
    llvm::Value* element, Type* element_type, llvm::Value* buffer,
    int64_t bit_offset) {
  switch (element_type->kind()) {
    case TypeKind::kBits:
      if (element->getType() != buffer->getType()) {
        element = builder_->CreateZExt(element, buffer->getType());
      }
      element = builder_->CreateShl(element, bit_offset);
      return builder_->CreateOr(buffer, element);
    case TypeKind::kArray: {
      ArrayType* array_type = element_type->AsArrayOrDie();
      Type* array_element_type = array_type->element_type();
      for (uint32_t i = 0; i < array_type->size(); i++) {
        XLS_ASSIGN_OR_RETURN(
            buffer, PackElement(builder_->CreateExtractValue(element, {i}),
                                array_element_type, buffer,
                                bit_offset +
                                    i * array_element_type->GetFlatBitCount()));
      }
      return buffer;
    }
    case TypeKind::kTuple: {
      // As with HandlePackedParam, we need to reverse tuple packing order to
      // match native layout.
      TupleType* tuple_type = element_type->AsTupleOrDie();
      for (int64_t i = tuple_type->size() - 1; i >= 0; i--) {
        XLS_ASSIGN_OR_RETURN(
            buffer,
            PackElement(builder_->CreateExtractValue(
                            element, {static_cast<uint32_t>(i)}),
                        tuple_type->element_type(i), buffer, bit_offset));
        bit_offset += tuple_type->element_type(i)->GetFlatBitCount();
      }
      return buffer;
    }
    case TypeKind::kToken: {
      // Tokens are zero-bit constructs, so there's nothing to do!
      return buffer;
    }
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unhandled element kind: ", TypeKindToString(element_type->kind())));
  }
}

void FunctionBuilderVisitor::UnpoisonOutputBuffer() {
#ifdef ABSL_HAVE_MEMORY_SANITIZER
  Type* xls_return_type = GetEffectiveReturnValue(xls_fn_)->GetType();
  llvm::ConstantInt* fn_addr =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx()),
                             absl::bit_cast<uint64_t>(&__msan_unpoison));
  llvm::Type* void_type = llvm::Type::getVoidTy(ctx());
  llvm::Type* u8_ptr_type =
      llvm::PointerType::get(llvm::Type::getInt8Ty(ctx()), /*AddressSpace=*/0);
  llvm::Type* size_t_type =
      llvm::Type::getIntNTy(ctx(), sizeof(size_t) * CHAR_BIT);
  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(void_type, {u8_ptr_type, size_t_type}, false);
  llvm::Value* fn_ptr =
      builder()->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));

  llvm::Value* out_param = GetOutputPtr();

  // The out_param can have any width, so cast the output argument pointer to a
  // u8 pointer to match the parameter types above.
  llvm::Value* out_param_casted =
      builder()->CreatePointerCast(out_param, u8_ptr_type);

  std::vector<llvm::Value*> args = {
      out_param_casted,
      llvm::ConstantInt::get(
          size_t_type, type_converter()->GetTypeByteSize(xls_return_type))};

  builder()->CreateCall(fn_type, fn_ptr, args);
#endif
}

/* static */ Node* FunctionBuilderVisitor::GetEffectiveReturnValue(
    FunctionBase* function_base) {
  if (function_base->IsFunction()) {
    return function_base->AsFunctionOrDie()->return_value();
  }
  XLS_CHECK(function_base->IsProc());
  return function_base->AsProcOrDie()->NextState();
}

}  // namespace xls
