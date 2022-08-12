// Copyright 2022 The XLS Authors
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
#include "xls/jit/ir_builder_visitor.h"

#include "llvm/include/llvm/IR/DerivedTypes.h"
#include "llvm/include/llvm/IR/Instructions.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/events.h"
#include "xls/ir/value_helpers.h"
#include "xls/jit/orc_jit.h"

#ifdef ABSL_HAVE_MEMORY_SANITIZER
#include <sanitizer/msan_interface.h>
#endif

#include "llvm/include/llvm/IR/Constants.h"
#include "xls/codegen/vast.h"
#include "xls/ir/function.h"
#include "xls/ir/proc.h"
#include "xls/jit/jit_runtime.h"

namespace xls {
namespace {

// Returns a sequence of numbered strings. Example: NumberedStrings("foo", 3)
// returns: {"foo_0", "foo_1", "foo_2"}
std::vector<std::string> NumberedStrings(absl::string_view s, int64_t count) {
  std::vector<std::string> result(count);
  for (int64_t i = 0; i < count; ++i) {
    result[i] = absl::StrFormat("%s_%d", s, i);
  }
  return result;
}

// Returns the concatentation of two vectors.
std::vector<std::string> ConcatVectors(absl::Span<const std::string> a,
                                       absl::Span<const std::string> b) {
  std::vector<std::string> result(a.begin(), a.end());
  result.insert(result.end(), b.begin(), b.end());
  return result;
}

// Emit an LLVM shift operation corresponding to the semantics of the given XLS
// op.
llvm::Value* EmitShiftOp(Node* shift, llvm::Value* lhs, llvm::Value* rhs,
                         llvm::IRBuilder<>* builder,
                         LlvmTypeConverter* type_converter) {
  Op op = shift->op();
  // Shift operands are allowed to be different sizes in the [XLS] IR, so
  // we need to cast them to be the same size here (for LLVM).
  int common_width = std::max(lhs->getType()->getIntegerBitWidth(),
                              rhs->getType()->getIntegerBitWidth());
  llvm::Type* dest_type =
      llvm::Type::getIntNTy(builder->getContext(), common_width);
  llvm::Value* wide_lhs =
      op == Op::kShra
          ? type_converter->AsSignedValue(lhs, shift->operand(0)->GetType(),
                                          *builder, dest_type)
          : builder->CreateZExt(lhs, dest_type);
  llvm::Value* wide_rhs = builder->CreateZExt(rhs, dest_type);
  // In LLVM, an overshifted shift creates poison.
  llvm::Value* is_overshift = builder->CreateICmpUGE(
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
  llvm::Value* safe_rhs = builder->CreateSelect(is_overshift, zero, wide_rhs);

  if (op == Op::kShll) {
    inst = builder->CreateShl(wide_lhs, safe_rhs);
  } else if (op == Op::kShra) {
    llvm::Value* high_bit = builder->CreateLShr(
        wide_lhs,
        llvm::ConstantInt::get(dest_type,
                               wide_lhs->getType()->getIntegerBitWidth() - 1));
    llvm::Value* high_bit_set =
        builder->CreateICmpEQ(high_bit, llvm::ConstantInt::get(dest_type, 1));
    overshift_value = builder->CreateSelect(
        high_bit_set, llvm::ConstantInt::getSigned(dest_type, -1), zero);
    inst = builder->CreateAShr(wide_lhs, safe_rhs);
  } else {
    XLS_CHECK_EQ(op, Op::kShrl);
    inst = builder->CreateLShr(wide_lhs, safe_rhs);
  }
  llvm::Value* result =
      builder->CreateSelect(is_overshift, overshift_value, inst);
  // The expected return type is the same as the original lhs. The shifted value
  // may be wider than this, so truncate in that case. This occurs if the
  // original rhs is wider than the original lhs.
  return result->getType() == lhs->getType()
             ? result
             : builder->CreateTrunc(result, lhs->getType());
}

llvm::Value* EmitDiv(llvm::Value* lhs, llvm::Value* rhs, bool is_signed,
                     LlvmTypeConverter* type_converter,
                     llvm::IRBuilder<>* builder) {
  // XLS div semantics differ from LLVM's (and most software's) here: in XLS,
  // division by zero returns the greatest value of that type, so 255 for an
  // unsigned byte, and either -128 or 127 for a signed one.
  // Thus, a little more work is necessary to emit LLVM IR matching the XLS
  // div op than just IRBuilder::Create[SU]Div().
  int type_width = rhs->getType()->getIntegerBitWidth();
  llvm::Value* zero = llvm::ConstantInt::get(rhs->getType(), 0);
  llvm::Value* rhs_eq_zero = builder->CreateICmpEQ(rhs, zero);
  llvm::Value* lhs_gt_zero = builder->CreateICmpSGT(lhs, zero);

  // If rhs is zero, make LHS = the max/min value and the RHS 1,
  // rather than introducing a proper conditional.
  rhs = builder->CreateSelect(rhs_eq_zero,
                              llvm::ConstantInt::get(rhs->getType(), 1), rhs);
  if (is_signed) {
    llvm::Value* max_value =
        type_converter
            ->ToLlvmConstant(rhs->getType(), Value(Bits::MaxSigned(type_width)))
            .value();
    llvm::Value* min_value =
        type_converter
            ->ToLlvmConstant(rhs->getType(), Value(Bits::MinSigned(type_width)))
            .value();

    lhs = builder->CreateSelect(
        rhs_eq_zero, builder->CreateSelect(lhs_gt_zero, max_value, min_value),
        lhs);
    return builder->CreateSDiv(lhs, rhs);
  }

  lhs = builder->CreateSelect(
      rhs_eq_zero,
      type_converter
          ->ToLlvmConstant(rhs->getType(), Value(Bits::AllOnes(type_width)))
          .value(),
      lhs);
  return builder->CreateUDiv(lhs, rhs);
}

llvm::Value* EmitMod(llvm::Value* lhs, llvm::Value* rhs, bool is_signed,
                     llvm::IRBuilder<>* builder) {
  // XLS mod semantics differ from LLVMs with regard to mod by zero. In XLS,
  // modulo by zero returns zero rather than undefined behavior.
  llvm::Value* zero = llvm::ConstantInt::get(rhs->getType(), 0);
  llvm::Value* rhs_eq_zero = builder->CreateICmpEQ(rhs, zero);
  // Replace a zero rhs with one to avoid SIGFPE even though the result is not
  // used.
  rhs = builder->CreateSelect(rhs_eq_zero,
                              llvm::ConstantInt::get(rhs->getType(), 1), rhs);
  return builder->CreateSelect(rhs_eq_zero, zero,
                               is_signed ? builder->CreateSRem(lhs, rhs)
                                         : builder->CreateURem(lhs, rhs));
}

// ORs together all elements in the two given values, be they Bits, Arrays, or
// Tuples.
llvm::Value* CreateAggregateOr(llvm::Value* lhs, llvm::Value* rhs,
                               llvm::IRBuilder<>* builder) {
  llvm::Type* arg_type = lhs->getType();
  if (arg_type->isIntegerTy()) {
    return builder->CreateOr(lhs, rhs);
  }

  llvm::Value* result = IrBuilderVisitor::CreateTypedZeroValue(arg_type);
  int num_elements = arg_type->isArrayTy() ? arg_type->getArrayNumElements()
                                           : arg_type->getNumContainedTypes();
  for (uint32_t i = 0; i < num_elements; ++i) {
    llvm::Value* iter_result =
        CreateAggregateOr(builder->CreateExtractValue(lhs, {i}),
                          builder->CreateExtractValue(rhs, {i}), builder);
    result = builder->CreateInsertValue(result, iter_result, {i});
  }

  return result;
}

// Local struct to hold the individual elements of a (possibly) compound
// comparison.
struct CompareTerm {
  llvm::Value* lhs;
  llvm::Value* rhs;
};

// Expand the lhs and rhs of a comparison into a vector of the individual leaf
// terms to compare.
absl::StatusOr<std::vector<CompareTerm>> ExpandTerms(
    Node* lhs, llvm::Value* llvm_lhs, Node* rhs, llvm::Value* llvm_rhs,
    Node* src, llvm::IRBuilder<>* builder) {
  XLS_RET_CHECK(lhs->GetType() == rhs->GetType()) << absl::StreamFormat(
      "The lhs and rhs of %s have different types: lhs %s rhs %s",
      src->ToString(), lhs->GetType()->ToString(), rhs->GetType()->ToString());

  struct ToExpand {
    Type* ty;
    llvm::Value* lhs;
    llvm::Value* rhs;
  };

  std::vector<ToExpand> unexpanded = {
      ToExpand{lhs->GetType(), llvm_lhs, llvm_rhs}};

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
          llvm::Value* lhs_value = builder->CreateExtractValue(next.lhs, {i});
          llvm::Value* rhs_value = builder->CreateExtractValue(next.rhs, {i});
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
          llvm::Value* lhs_value = builder->CreateExtractValue(next.lhs, {i});
          llvm::Value* rhs_value = builder->CreateExtractValue(next.rhs, {i});
          unexpanded.push_back(ToExpand{element_type, lhs_value, rhs_value});
        }
        break;
      }
    }
  }
  return terms;
}

// Returns the result of indexing into 'array' using the scalar index value
// 'index'. 'array_size' is the number of elements in the array.
absl::StatusOr<llvm::Value*> IndexIntoArray(llvm::Value* array,
                                            llvm::Value* index,
                                            int64_t array_size,
                                            LlvmTypeConverter* type_converter,
                                            llvm::IRBuilder<>* builder) {
  int64_t index_width = index->getType()->getIntegerBitWidth();

  // Check for out-of-bounds access. If the index is out of bounds it is set to
  // the maximum index value.
  int64_t index_bitwidth = index->getType()->getIntegerBitWidth();
  int64_t comparison_bitwidth = std::max(index_bitwidth, int64_t{64});
  llvm::Value* array_size_comparison_bitwidth = llvm::ConstantInt::get(
      llvm::Type::getIntNTy(builder->getContext(), comparison_bitwidth),
      array_size);
  llvm::Value* index_value_comparison_bitwidth = builder->CreateZExt(
      index, llvm::Type::getIntNTy(builder->getContext(), comparison_bitwidth));
  llvm::Value* is_index_inbounds = builder->CreateICmpULT(
      index_value_comparison_bitwidth, array_size_comparison_bitwidth);
  llvm::Value* inbounds_index = builder->CreateSelect(
      is_index_inbounds, index,
      llvm::ConstantInt::get(index->getType(), array_size - 1));

  // Our IR does not use negative indices, so we add a
  // zero MSb to prevent LLVM from interpreting this as such.
  std::vector<llvm::Value*> gep_indices = {
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(builder->getContext()), 0),
      builder->CreateZExt(inbounds_index,
                          type_converter->GetLlvmBitsType(index_width + 1))};

  llvm::Type* array_type = array->getType();
  // Ideally, we'd use IRBuilder::CreateExtractValue here, but that requires
  // constant indices. Since there's no other way to extract a value from an
  // aggregate, we're left with storing the value in a temporary alloca and
  // using that pointer to extract the value.
  llvm::Value* alloca = builder->CreateAlloca(array_type);
  builder->CreateStore(array, alloca);

  llvm::Type* element_type = array_type->getArrayElementType();
  llvm::Value* gep = builder->CreateGEP(array_type, alloca, gep_indices);
  return builder->CreateLoad(element_type, gep);
}

// This is a shim to let JIT code add a new trace fragment to an existing trace
// buffer.
void PerformStringStep(char* step_string, std::string* buffer) {
  buffer->append(step_string);
}

// Build the LLVM IR that handles string fragment format steps.
absl::Status InvokeStringStepCallback(llvm::IRBuilder<>* builder,
                                      const std::string& step_string,
                                      llvm::Value* buffer_ptr) {
  llvm::Constant* step_constant = builder->CreateGlobalStringPtr(step_string);

  std::vector<llvm::Type*> params = {step_constant->getType(),
                                     buffer_ptr->getType()};

  llvm::Type* void_type = llvm::Type::getVoidTy(builder->getContext());

  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(void_type, params, /*isVarArg=*/false);

  std::vector<llvm::Value*> args = {step_constant, buffer_ptr};

  llvm::ConstantInt* fn_addr =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(builder->getContext()),
                             absl::bit_cast<uint64_t>(&PerformStringStep));
  llvm::Value* fn_ptr =
      builder->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));
  builder->CreateCall(fn_type, fn_ptr, args);
  return absl::OkStatus();
}

void PerformFormatStep(JitRuntime* runtime, const xls::Type* type,
                       const uint8_t* value, uint64_t format_u64,
                       std::string* buffer) {
  FormatPreference format = static_cast<FormatPreference>(format_u64);
  Value ir_value = runtime->UnpackBuffer(value, type, /*unpoison=*/true);
  absl::StrAppend(buffer, ir_value.ToHumanString(format));
}

// Build the LLVM IR that handles formatting a runtime value according to a
// format preference.
absl::Status InvokeFormatStepCallback(llvm::IRBuilder<>* builder,
                                      FormatPreference format,
                                      xls::Type* operand_type,
                                      llvm::Value* operand,
                                      llvm::Value* buffer_ptr,
                                      llvm::Value* jit_runtime_ptr) {
  llvm::Type* void_type = llvm::Type::getVoidTy(builder->getContext());
  auto* i64_type = llvm::Type::getInt64Ty(builder->getContext());

  llvm::ConstantInt* llvm_format =
      llvm::ConstantInt::get(i64_type, static_cast<uint64_t>(format));

  // Note: we assume the package lifetime is >= that of the JIT code by
  // capturing this type pointer as a value burned into the JIT code, which
  // should always be true.
  llvm::ConstantInt* llvm_operand_type =
      llvm::ConstantInt::get(i64_type, absl::bit_cast<uint64_t>(operand_type));

  std::vector<llvm::Type*> params = {
      jit_runtime_ptr->getType(), llvm_operand_type->getType(),
      operand->getType(), llvm_format->getType(), buffer_ptr->getType()};
  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(void_type, params, /*isVarArg=*/false);

  std::vector<llvm::Value*> args = {jit_runtime_ptr, llvm_operand_type, operand,
                                    llvm_format, buffer_ptr};

  llvm::ConstantInt* fn_addr = llvm::ConstantInt::get(
      i64_type, absl::bit_cast<uint64_t>(&PerformFormatStep));
  llvm::Value* fn_ptr =
      builder->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));
  builder->CreateCall(fn_type, fn_ptr, args);
  return absl::OkStatus();
}

// This a shim to let JIT code record a completed trace as an interpreter event.
void RecordTrace(std::string* buffer, xls::InterpreterEvents* events) {
  events->trace_msgs.push_back(*buffer);
  delete buffer;
}

// Build the LLVM IR to invoke the callback that records traces.
absl::Status InvokeRecordTraceCallback(llvm::IRBuilder<>* builder,
                                       llvm::Value* buffer_ptr,
                                       llvm::Value* interpreter_events_ptr) {
  llvm::Type* ptr_type = llvm::PointerType::get(builder->getContext(), 0);

  std::vector<llvm::Type*> params = {buffer_ptr->getType(), ptr_type};

  llvm::Type* void_type = llvm::Type::getVoidTy(builder->getContext());

  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(void_type, params, /*isVarArg=*/false);

  std::vector<llvm::Value*> args = {buffer_ptr, interpreter_events_ptr};

  llvm::ConstantInt* fn_addr =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(builder->getContext()),
                             absl::bit_cast<uint64_t>(&RecordTrace));
  llvm::Value* fn_ptr =
      builder->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));
  builder->CreateCall(fn_type, fn_ptr, args);
  return absl::OkStatus();
}

// This is a shim to let JIT code create a buffer for accumulating trace
// fragments.
std::string* CreateTraceBuffer() { return new std::string(); }

// Build the LLVM IR to invoke the callback that creates a trace buffer.
absl::StatusOr<llvm::Value*> InvokeCreateBufferCallback(
    llvm::IRBuilder<>* builder) {
  std::vector<llvm::Type*> params;

  llvm::Type* ptr_type = llvm::PointerType::get(builder->getContext(), 0);

  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(ptr_type, params, /*isVarArg=*/false);

  std::vector<llvm::Value*> args;

  llvm::ConstantInt* fn_addr =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(builder->getContext()),
                             absl::bit_cast<uint64_t>(&CreateTraceBuffer));
  llvm::Value* fn_ptr =
      builder->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));
  return builder->CreateCall(fn_type, fn_ptr, args);
}

// This a shim to let JIT code record an assertion failure as an interpreter
// event.
void RecordAssertion(char* msg, xls::InterpreterEvents* events) {
  events->assert_msgs.push_back(msg);
}

// Build the LLVM IR to invoke the callback that records assertions.
absl::Status InvokeAssertCallback(llvm::IRBuilder<>* builder,
                                  const std::string& message,
                                  llvm::Value* interpreter_events_ptr) {
  llvm::Constant* msg_constant = builder->CreateGlobalStringPtr(message);

  llvm::Type* msg_type = msg_constant->getType();

  llvm::Type* ptr_type = llvm::PointerType::get(builder->getContext(), 0);

  std::vector<llvm::Type*> params = {msg_type, ptr_type};

  llvm::Type* void_type = llvm::Type::getVoidTy(builder->getContext());

  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(void_type, params, /*isVarArg=*/false);

  std::vector<llvm::Value*> args = {msg_constant, interpreter_events_ptr};

  llvm::ConstantInt* fn_addr =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(builder->getContext()),
                             absl::bit_cast<uint64_t>(&RecordAssertion));
  llvm::Value* fn_ptr =
      builder->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));
  builder->CreateCall(fn_type, fn_ptr, args);
  return absl::OkStatus();
}

}  // namespace

llvm::Function* NodeIrContext::CreateFunction(
    Node* node, LlvmTypeConverter* type_converter, llvm::Module* module,
    std::optional<Environment> environment) {
  std::vector<llvm::Type*> param_types;
  for (Node* operand : node->operands()) {
    param_types.push_back(
        type_converter->ConvertToLlvmType(operand->GetType()));
  }
  if (environment.has_value()) {
    // Add parameters for events, runtime, and user_data.
    llvm::PointerType* ptr_type =
        llvm::PointerType::get(module->getContext(), 0);
    param_types.push_back(ptr_type);
    param_types.push_back(ptr_type);
    param_types.push_back(ptr_type);
  }
  llvm::Type* result_type = type_converter->ConvertToLlvmType(node->GetType());
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      result_type,
      llvm::ArrayRef<llvm::Type*>(param_types.data(), param_types.size()),
      /*isVarArg=*/false);

  // To avoid name collisions between nodes from other functions append the
  // unique ID to the function name.
  std::string function_name =
      absl::StrFormat("__%s_%d", node->GetName(), node->id());
  llvm::Function* llvm_function = llvm::cast<llvm::Function>(
      module->getOrInsertFunction(function_name, function_type).getCallee());

  // Mark as private so function can be deleted after inlining.
  llvm_function->setLinkage(llvm::GlobalValue::PrivateLinkage);

  return llvm_function;
}

absl::StatusOr<NodeIrContext> NodeIrContext::Create(
    Node* node, absl::Span<const std::string> operand_names,
    LlvmTypeConverter* type_converter, llvm::Module* module,
    std::optional<Environment> environment) {
  NodeIrContext nc;
  nc.node_ = node;
  nc.llvm_function_ = CreateFunction(node, type_converter, module, environment);
  nc.type_converter_ = type_converter;

  nc.builder_ = std::make_unique<llvm::IRBuilder<>>(
      llvm::BasicBlock::Create(module->getContext(), "entry", nc.llvm_function_,
                               /*InsertBefore=*/nullptr));

  XLS_RET_CHECK_EQ(operand_names.size(), node->operands().size());
  for (int64_t i = 0; i < node->operand_count(); ++i) {
    nc.operands_.push_back(nc.llvm_function_->getArg(i));
    nc.operands_.back()->setName(operand_names[i]);
  }

  if (environment.has_value()) {
    nc.environment_ = Environment();
    nc.environment_->events = nc.llvm_function_->getArg(node->operand_count());
    nc.environment_->events->setName("events");
    nc.environment_->user_data =
        nc.llvm_function_->getArg(node->operand_count() + 1);
    nc.environment_->user_data->setName("user_data");
    nc.environment_->jit_runtime =
        nc.llvm_function_->getArg(node->operand_count() + 2);
    nc.environment_->jit_runtime->setName("jit_runtime");
  }

  return nc;
}

void NodeIrContext::Finalize(llvm::Value* result,
                             std::optional<llvm::IRBuilder<>*> exit_builder) {
  llvm::IRBuilder<>* b =
      exit_builder.has_value() ? exit_builder.value() : &builder();
  result = type_converter()->ClearPaddingBits(result, node()->GetType(), *b);
  b->CreateRet(result);
}

IrBuilderVisitor::IrBuilderVisitor(
    llvm::Function* llvm_fn, FunctionBase* xls_fn,
    LlvmTypeConverter* type_converter,
    std::function<absl::StatusOr<llvm::Function*>(Function*)> function_builder)
    : dispatch_fn_(llvm_fn),
      xls_fn_(xls_fn),
      dispatch_builder_(std::make_unique<llvm::IRBuilder<>>(
          llvm::BasicBlock::Create(llvm_fn->getContext(), "entry", llvm_fn,
                                   /*InsertBefore=*/nullptr))),
      type_converter_(type_converter),
      function_builder_(function_builder) {}

absl::Status IrBuilderVisitor::DefaultHandler(Node* node) {
  return absl::UnimplementedError(
      absl::StrCat("Unhandled node: ", node->ToString()));
}

absl::Status IrBuilderVisitor::HandleAdd(BinOp* binop) {
  return HandleBinaryOp(
      binop, [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateAdd(lhs, rhs);
      });
}

absl::Status IrBuilderVisitor::HandleAndReduce(BitwiseReductionOp* op) {
  return HandleUnaryOp(op, [&](llvm::Value* operand, llvm::IRBuilder<>& b) {
    // AND-reduce is equivalent to checking if every bit is set in the
    // input.
    return b.CreateICmpEQ(
        operand, type_converter()->PaddingMask(op->operand(0)->GetType(), b));
  });
}

absl::Status IrBuilderVisitor::HandleAfterAll(AfterAll* after_all) {
  // AfterAll is only meaningful to the compiler and does not actually perform
  // any computation.
  return HandleNaryOp(after_all, [&](absl::Span<llvm::Value* const> operands,
                                     llvm::IRBuilder<>& b) {
    return type_converter_->GetToken();
  });
}

absl::Status IrBuilderVisitor::HandleAssert(Assert* assert_op) {
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(assert_op, {"tkn", "condition"},
                                        /*include_environment=*/true));

  llvm::IRBuilder<>& b = node_context.builder();
  llvm::Function* function = node_context.llvm_function();

  std::string assert_label = assert_op->label().value_or("assert");

  llvm::BasicBlock* after_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(assert_label, "_after"), function);

  llvm::BasicBlock* ok_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(assert_label, "_ok"), function);

  llvm::IRBuilder<> ok_builder(ok_block);

  ok_builder.CreateBr(after_block);

  llvm::BasicBlock* fail_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(assert_label, "_fail"), function);
  llvm::IRBuilder<> fail_builder(fail_block);
  XLS_RETURN_IF_ERROR(
      InvokeAssertCallback(&fail_builder, assert_op->message(),
                           node_context.GetInterpreterEvents()));

  fail_builder.CreateBr(after_block);

  b.CreateCondBr(node_context.operand(1), ok_block, fail_block);

  auto after_builder = std::make_unique<llvm::IRBuilder<>>(after_block);
  llvm::Value* token = type_converter_->GetToken();
  return FinalizeNodeIrContext(node_context, token, std::move(after_builder));
}

absl::Status IrBuilderVisitor::HandleArray(Array* array) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(array,
                       NumberedStrings("operand", array->operand_count())));
  llvm::IRBuilder<>& b = node_context.builder();

  llvm::Type* array_type = type_converter_->ConvertToLlvmType(array->GetType());

  llvm::Value* result = CreateTypedZeroValue(array_type);
  for (uint32_t i = 0; i < array->size(); ++i) {
    result = b.CreateInsertValue(result, node_context.operand(i), {i});
  }

  return FinalizeNodeIrContext(node_context, result);
}

absl::Status IrBuilderVisitor::HandleTrace(Trace* trace_op) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(
          trace_op,
          ConcatVectors({"tkn", "condition"},
                        NumberedStrings("arg", trace_op->args().size())),
          /*include_environment=*/true));

  llvm::IRBuilder<>& b = node_context.builder();
  llvm::Value* condition = node_context.operand(1);
  llvm::Value* events_ptr = node_context.GetInterpreterEvents();
  llvm::Value* jit_runtime_ptr = node_context.GetJitRuntime();

  std::string trace_name = trace_op->GetName();

  llvm::BasicBlock* after_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(trace_name, "_after"), node_context.llvm_function());

  llvm::BasicBlock* skip_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(trace_name, "_skip"), node_context.llvm_function());

  llvm::IRBuilder<> skip_builder(skip_block);

  skip_builder.CreateBr(after_block);

  llvm::BasicBlock* print_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(trace_name, "_print"), node_context.llvm_function());
  llvm::IRBuilder<> print_builder(print_block);

  XLS_ASSIGN_OR_RETURN(llvm::Value * buffer_ptr,
                       InvokeCreateBufferCallback(&print_builder));

  // Operands are: (tok, pred, ..data_operands..)
  XLS_RET_CHECK_EQ(trace_op->operand(0)->GetType(),
                   trace_op->package()->GetTokenType());
  XLS_RET_CHECK_EQ(trace_op->operand(1)->GetType(),
                   trace_op->package()->GetBitsType(1));

  size_t operand_index = 2;
  for (const FormatStep& step : trace_op->format()) {
    if (absl::holds_alternative<std::string>(step)) {
      XLS_RETURN_IF_ERROR(InvokeStringStepCallback(
          &print_builder, absl::get<std::string>(step), buffer_ptr));
    } else {
      xls::Node* o = trace_op->operand(operand_index);
      llvm::Value* operand = node_context.operand(operand_index);
      llvm::AllocaInst* alloca = print_builder.CreateAlloca(operand->getType());
      print_builder.CreateStore(operand, alloca);
      // The way our format strings are currently formed we implicitly refer to
      // the next operand after formatting this one.
      operand_index += 1;
      XLS_RETURN_IF_ERROR(InvokeFormatStepCallback(
          &print_builder, absl::get<FormatPreference>(step), o->GetType(),
          alloca, buffer_ptr, jit_runtime_ptr));
    }
  }

  XLS_RETURN_IF_ERROR(
      InvokeRecordTraceCallback(&print_builder, buffer_ptr, events_ptr));

  print_builder.CreateBr(after_block);

  b.CreateCondBr(condition, print_block, skip_block);

  auto after_builder = std::make_unique<llvm::IRBuilder<>>(after_block);
  llvm::Value* token = type_converter_->GetToken();
  return FinalizeNodeIrContext(node_context, token, std::move(after_builder));
}

absl::Status IrBuilderVisitor::HandleArrayIndex(ArrayIndex* index) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(
          index,
          ConcatVectors({"array"},
                        NumberedStrings("index", index->indices().size()))));
  llvm::IRBuilder<>& b = node_context.builder();

  Type* element_type = index->array()->GetType();
  llvm::Value* element = node_context.operand(0);
  // Index operands start at 1.
  for (int64_t i = 1; i < index->operand_count(); ++i) {
    llvm::Value* index_value = node_context.operand(i);
    XLS_ASSIGN_OR_RETURN(element,
                         IndexIntoArray(element, index_value,
                                        element_type->AsArrayOrDie()->size(),
                                        type_converter(), &b));
    element_type = element_type->AsArrayOrDie()->element_type();
  }
  return FinalizeNodeIrContext(node_context, element);
}

absl::Status IrBuilderVisitor::HandleArraySlice(ArraySlice* slice) {
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(slice, {"array", "start"}));
  llvm::IRBuilder<>& b = node_context.builder();

  llvm::Value* array = node_context.operand(0);
  llvm::Value* start = node_context.operand(1);
  int64_t width = slice->width();

  // This overestimates the number of bits needed but in all practical
  // situations it should be fine. The only exception to that is if some code
  // uses a 64 bit index but doesn't actually make full use of that range, then
  // this will possibly push us over a performance cliff.
  int64_t index_bits = start->getType()->getIntegerBitWidth() +
                       Bits::MinBitCountSigned(width) + 1;
  llvm::Type* index_type = type_converter()->ConvertToLlvmType(
      slice->package()->GetBitsType(index_bits));
  llvm::Type* result_type =
      type_converter_->ConvertToLlvmType(slice->GetType());
  llvm::Type* result_element_type = type_converter_->ConvertToLlvmType(
      slice->GetType()->AsArrayOrDie()->element_type());
  llvm::Value* alloca = b.CreateAlloca(result_type, 0, "alloca");
  llvm::Value* start_big = b.CreateZExt(start, index_type, "start_big");

  for (int64_t i = 0; i < width; i++) {
    llvm::Value* index =
        b.CreateAdd(start_big, llvm::ConstantInt::get(index_type, i), "index");
    XLS_ASSIGN_OR_RETURN(
        llvm::Value * value,
        IndexIntoArray(array, index,
                       slice->array()->GetType()->AsArrayOrDie()->size(),
                       type_converter(), &b));
    std::vector<llvm::Value*> gep_indices = {
        llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx()), i)};
    llvm::Value* gep = b.CreateGEP(result_element_type, alloca, gep_indices);
    b.CreateStore(value, gep);
  }

  llvm::Value* sliced_array = b.CreateLoad(result_type, alloca);

  return FinalizeNodeIrContext(node_context, sliced_array);
}

absl::Status IrBuilderVisitor::HandleArrayUpdate(ArrayUpdate* update) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(
          update,
          ConcatVectors({"array", "update_value"},
                        NumberedStrings("index", update->indices().size()))));
  llvm::IRBuilder<>& b = node_context.builder();
  llvm::Value* original_array = node_context.operand(0);
  llvm::Value* update_value = node_context.operand(1);
  auto indices = node_context.operands().subspan(2);

  if (indices.empty()) {
    // An empty index replaces the entire array value.
    return FinalizeNodeIrContext(node_context, update_value);
  }

  llvm::Type* array_type = original_array->getType();
  llvm::AllocaInst* alloca = b.CreateAlloca(array_type);
  b.CreateStore(original_array, alloca);

  Type* element_type = update->array_to_update()->GetType();
  std::vector<llvm::Value*> gep_indices = {
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx()), 0)};
  llvm::Value* is_inbounds = b.getTrue();
  for (llvm::Value* index : indices) {
    int64_t index_bitwidth = index->getType()->getIntegerBitWidth();
    int64_t comparison_bitwidth = std::max(index_bitwidth, int64_t{64});
    llvm::Value* array_size_comparison_bitwidth = llvm::ConstantInt::get(
        b.getIntNTy(comparison_bitwidth), element_type->AsArrayOrDie()->size());
    llvm::Value* index_value_comparison_bitwidth =
        b.CreateZExt(index, llvm::Type::getIntNTy(ctx(), comparison_bitwidth));
    llvm::Value* is_index_inbounds =
        b.CreateICmpULT(index_value_comparison_bitwidth,
                        array_size_comparison_bitwidth, "idx_is_inbounds");

    gep_indices.push_back(index_value_comparison_bitwidth);
    is_inbounds = b.CreateAnd(is_inbounds, is_index_inbounds, "inbounds");

    element_type = element_type->AsArrayOrDie()->element_type();
  }

  // Create the join block which occurs after the conditional block (conditioned
  // on whether the index is inbounds).
  llvm::BasicBlock* join_block =
      llvm::BasicBlock::Create(ctx(), absl::StrCat(update->GetName(), "_join"),
                               node_context.llvm_function());

  // Create the inbounds block and fill with a store to the array elemnt.
  llvm::BasicBlock* inbounds_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(update->GetName(), "_inbounds"),
      node_context.llvm_function(),
      /*InsertBefore=*/join_block);
  llvm::IRBuilder<> inbounds_builder(inbounds_block);
  llvm::Value* gep =
      inbounds_builder.CreateGEP(array_type, alloca, gep_indices);
  inbounds_builder.CreateStore(update_value, gep);
  inbounds_builder.CreateBr(join_block);

  // Create a conditional branch using the original builder (end of the BB
  // before the if/then).
  b.CreateCondBr(is_inbounds, inbounds_block, join_block);

  // Create a new BB at the join point.
  auto exit_builder = std::make_unique<llvm::IRBuilder<>>(join_block);
  llvm::Value* update_array = exit_builder->CreateLoad(array_type, alloca);

  return FinalizeNodeIrContext(node_context, update_array,
                               std::move(exit_builder));
}

absl::Status IrBuilderVisitor::HandleArrayConcat(ArrayConcat* concat) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(concat,
                       NumberedStrings("operand", concat->operand_count())));
  llvm::IRBuilder<>& b = node_context.builder();

  llvm::Type* array_type =
      type_converter_->ConvertToLlvmType(concat->GetType());

  llvm::Value* result = CreateTypedZeroValue(array_type);

  int64_t result_index = 0;
  int64_t result_elements = array_type->getArrayNumElements();
  for (llvm::Value* array : node_context.operands()) {
    llvm::Type* array_type = array->getType();

    int64_t element_count = array_type->getArrayNumElements();
    for (int64_t j = 0; j < element_count; ++j) {
      llvm::Value* element =
          b.CreateExtractValue(array, {static_cast<uint32_t>(j)});

      if (result_index >= result_elements) {
        return absl::InternalError(absl::StrFormat(
            "array-concat %s result and source have mismatched number of "
            "elements - expected %d",
            concat->ToString(), result_elements));
      }

      result = b.CreateInsertValue(result, element,
                                   {static_cast<uint32_t>(result_index)});
      ++result_index;
    }
  }

  return FinalizeNodeIrContext(node_context, result);
}

absl::Status IrBuilderVisitor::HandleBitSlice(BitSlice* bit_slice) {
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(bit_slice, {"operand"}));
  llvm::IRBuilder<>& b = node_context.builder();
  llvm::Value* value = node_context.operand(0);
  Value shift_amount(
      UBits(bit_slice->start(), value->getType()->getIntegerBitWidth()));
  XLS_ASSIGN_OR_RETURN(
      llvm::Constant * start,
      type_converter_->ToLlvmConstant(value->getType(), shift_amount));

  // Then shift and "mask" (by casting) the input value.
  llvm::Value* shifted_value = b.CreateLShr(value, start);
  llvm::Value* truncated_value = b.CreateTrunc(
      shifted_value, type_converter()->ConvertToLlvmType(bit_slice->GetType()));

  return FinalizeNodeIrContext(node_context, truncated_value);
}

absl::Status IrBuilderVisitor::HandleBitSliceUpdate(BitSliceUpdate* update) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(update, {"to_update", "start", "update_value"}));
  llvm::IRBuilder<>& b = node_context.builder();
  llvm::Value* to_update = node_context.operand(0);
  llvm::Value* start = node_context.operand(1);
  llvm::Value* update_value = node_context.operand(2);

  llvm::Type* result_type =
      type_converter()->ConvertToLlvmType(update->GetType());

  // Zero extend each value to the max of any of the values' widths.
  int max_width = std::max(std::max(to_update->getType()->getIntegerBitWidth(),
                                    start->getType()->getIntegerBitWidth()),
                           update_value->getType()->getIntegerBitWidth());
  llvm::IntegerType* max_width_type = b.getIntNTy(max_width);
  llvm::Value* to_update_wide =
      b.CreateZExt(to_update, max_width_type, "to_update");
  llvm::Value* start_wide = b.CreateZExt(start, max_width_type, "start");
  llvm::Value* update_value_wide =
      b.CreateZExt(update_value, max_width_type, "update_value");

  // If start is greater than or equal to the width of to_update, then the
  // updated slice is entirely out of bounds the result of the operation is
  // simply to_update.
  llvm::Value* in_bounds =
      b.CreateICmpULT(start_wide,
                      llvm::ConstantInt::get(
                          max_width_type, update->operand(0)->BitCountOrDie()),
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
          max_width_type,
          Value(bits_ops::ZeroExtend(
              Bits::AllOnes(update->update_value()->BitCountOrDie()),
              max_width))));
  // Set the shift amount to the 0 in the case of overshift (start >= max_width)
  // to avoid creating a poisonous shift value which can run afoul of LLVM
  // optimization bugs. The shifted value is not even used in this case.
  llvm::Value* shift_amount = b.CreateSelect(
      in_bounds, start_wide, llvm::ConstantInt::get(max_width_type, 0));
  llvm::Value* shifted_mask =
      b.CreateNot(b.CreateShl(mask, shift_amount), "mask");
  llvm::Value* masked_to_update = b.CreateAnd(shifted_mask, to_update_wide);
  llvm::Value* shifted_update_value =
      b.CreateShl(update_value_wide, shift_amount);
  llvm::Value* updated_slice =
      b.CreateTrunc(b.CreateOr(masked_to_update, shifted_update_value),
                    result_type, "updated_slice");

  llvm::Value* result =
      b.CreateSelect(in_bounds, updated_slice, to_update, "result");

  return FinalizeNodeIrContext(node_context, result);
}

absl::Status IrBuilderVisitor::HandleDynamicBitSlice(
    DynamicBitSlice* dynamic_bit_slice) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(dynamic_bit_slice, {"operand", "start"}));
  llvm::IRBuilder<>& b = node_context.builder();
  llvm::Value* value = node_context.operand(0);
  llvm::Value* start = node_context.operand(1);

  int64_t value_width = value->getType()->getIntegerBitWidth();
  int64_t start_width = start->getType()->getIntegerBitWidth();
  // Either value or start may be wider, so we use the widest of both
  // since LLVM requires both arguments to be of the same type for
  // comparison and shifting.
  int64_t max_width = std::max(start_width, value_width);
  llvm::IntegerType* max_width_type = b.getIntNTy(max_width);
  llvm::Value* value_ext = b.CreateZExt(value, max_width_type);
  llvm::Value* start_ext = b.CreateZExt(start, max_width_type);

  Value operand_width(UBits(value_width, max_width));
  XLS_ASSIGN_OR_RETURN(
      llvm::Constant * bit_width,
      type_converter_->ToLlvmConstant(max_width_type, operand_width));

  // "out_of_bounds" indicates whether slice is completely out of bounds.
  llvm::Value* out_of_bounds = b.CreateICmpUGE(start_ext, bit_width);
  llvm::Type* return_type =
      type_converter_->ConvertToLlvmType(dynamic_bit_slice->GetType());
  XLS_ASSIGN_OR_RETURN(
      llvm::Constant * zeros,
      type_converter_->ToLlvmConstant(return_type,
                                      Value(Bits(dynamic_bit_slice->width()))));
  // Then shift and truncate the input value. Set the shift amount to the 0 in
  // the case of overshift to avoid creating a poisonous shift value which can
  // run afoul of LLVM optimization bugs. The shifted value is not even used in
  // this case.
  llvm::Value* shift_amount = b.CreateSelect(
      out_of_bounds, llvm::ConstantInt::get(max_width_type, 0), start_ext);
  llvm::Value* shifted_value = b.CreateLShr(value_ext, shift_amount);
  llvm::Value* truncated_value = b.CreateTrunc(shifted_value, return_type);
  llvm::Value* result = b.CreateSelect(out_of_bounds, zeros, truncated_value);

  return FinalizeNodeIrContext(node_context, result);
}

absl::Status IrBuilderVisitor::HandleConcat(Concat* concat) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(concat,
                       NumberedStrings("operand", concat->operand_count())));
  llvm::IRBuilder<>& b = node_context.builder();

  llvm::Type* dest_type = type_converter_->ConvertToLlvmType(concat->GetType());
  llvm::Value* base = llvm::ConstantInt::get(dest_type, 0);

  int current_shift = concat->BitCountOrDie();
  for (int64_t i = 0; i < concat->operand_count(); ++i) {
    Node* xls_operand = concat->operand(i);
    llvm::Value* operand = node_context.operand(i);

    // Widen each operand to the full size, shift to the right location, and
    // bitwise or into the result value.
    int64_t operand_width = xls_operand->BitCountOrDie();
    operand = b.CreateZExt(operand, dest_type);
    llvm::Value* shifted_operand =
        b.CreateShl(operand, current_shift - operand_width);
    base = b.CreateOr(base, shifted_operand);

    current_shift -= operand_width;
  }

  return FinalizeNodeIrContext(node_context, base);
}

absl::Status IrBuilderVisitor::HandleCountedFor(CountedFor* counted_for) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(
          counted_for,
          ConcatVectors({"initial_value"},
                        NumberedStrings("invariant_arg",
                                        counted_for->invariant_args().size())),
          /*include_environment=*/true));
  llvm::IRBuilder<>& b = node_context.builder();

  XLS_ASSIGN_OR_RETURN(llvm::Function * function,
                       GetOrBuildFunction(counted_for->body()));
  // Add the loop carry and the index to the invariant arguments.
  std::vector<llvm::Value*> args(counted_for->invariant_args().size() + 2);
  for (int i = 0; i < counted_for->invariant_args().size(); i++) {
    // Invariant args start at operand 1 of the counted-for.
    args[i + 2] = node_context.operand(i + 1);
  }

  // Initial value is the zero-th operand of the counted-for.
  args[1] = node_context.operand(0);

  // Pass in the events, user_data, and runtime pointer to the loop body.
  args.push_back(node_context.GetInterpreterEvents());
  args.push_back(node_context.GetUserData());
  args.push_back(node_context.GetJitRuntime());

  llvm::Type* function_type = function->getFunctionType();
  for (int i = 0; i < counted_for->trip_count(); ++i) {
    args[0] = llvm::ConstantInt::get(function_type->getFunctionParamType(0),
                                     i * counted_for->stride());
    args[1] = b.CreateCall(function, args);
  }

  return FinalizeNodeIrContext(node_context, args[1]);
}

absl::Status IrBuilderVisitor::HandleCover(Cover* cover) {
  // TODO(https://github.com/google/xls/issues/499): 2021-09-17: Add coverpoint
  // support to the JIT.
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(cover, {"tkn", "condition"}));
  llvm::Value* token = type_converter_->GetToken();
  return FinalizeNodeIrContext(node_context, token);
}

absl::Status IrBuilderVisitor::HandleDecode(Decode* decode) {
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(decode, {"operand"}));
  llvm::IRBuilder<>& b = node_context.builder();
  llvm::Value* input = node_context.operand(0);

  llvm::Type* result_type =
      type_converter_->ConvertToLlvmType(decode->GetType());
  // If the input value is greater than this op's width, then return 0.
  // In that case, the shl will produce a poison value, but it'll be unused.
  llvm::Value* cast_input = b.CreateZExt(input, result_type);
  llvm::Value* overflow = b.CreateICmpUGE(
      cast_input, llvm::ConstantInt::get(result_type, decode->width()));
  llvm::Value* result = b.CreateSelect(
      overflow, llvm::ConstantInt::get(result_type, 0),
      b.CreateShl(llvm::ConstantInt::get(result_type, 1), cast_input));

  return FinalizeNodeIrContext(node_context, result);
}

absl::Status IrBuilderVisitor::HandleDynamicCountedFor(
    DynamicCountedFor* dynamic_counted_for) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(
          dynamic_counted_for,
          ConcatVectors(
              {"initial_value", "trip_count", "stride"},
              NumberedStrings("invariant_arg",
                              dynamic_counted_for->invariant_args().size())),
          /*include_environment=*/true));
  llvm::IRBuilder<>& b = node_context.builder();
  llvm::Value* initial_value = node_context.operand(0);
  llvm::Value* trip_count = node_context.operand(1);
  llvm::Value* stride = node_context.operand(2);
  auto invariant_args = node_context.operands().subspan(3);

  // Grab loop body.
  XLS_ASSIGN_OR_RETURN(llvm::Function * loop_body_function,
                       GetOrBuildFunction(dynamic_counted_for->body()));
  llvm::Type* loop_body_function_type = loop_body_function->getFunctionType();

  // The loop body arguments are the invariant arguments plus the loop carry and
  // index.
  std::vector<llvm::Value*> args(dynamic_counted_for->invariant_args().size() +
                                 2);
  for (int i = 0; i < invariant_args.size(); i++) {
    args[i + 2] = invariant_args[i];
  }

  // Pass in the events, user_data, and runtime pointer to the loop body.
  args.push_back(node_context.GetInterpreterEvents());
  args.push_back(node_context.GetUserData());
  args.push_back(node_context.GetJitRuntime());

  // Create basic blocks and corresponding builders. We have 4 blocks:
  // Entry     - the code executed before the loop.
  // Preheader - checks the loop condition.
  // Loop      - calls the loop body, updates the loop carry and index.
  // Exit      - the block we jump to after the loop.
  // ------------------------------------------------------------------
  // Entry
  llvm::BasicBlock* entry_block = b.GetInsertBlock();
  // Preheader
  llvm::BasicBlock* preheader_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(dynamic_counted_for->GetName(), "_preheader"),
      node_context.llvm_function());
  std::unique_ptr<llvm::IRBuilder<>> preheader_builder =
      std::make_unique<llvm::IRBuilder<>>(preheader_block);
  // Loop
  llvm::BasicBlock* loop_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(dynamic_counted_for->GetName(), "_loop_block"),
      node_context.llvm_function());
  std::unique_ptr<llvm::IRBuilder<>> loop_builder =
      std::make_unique<llvm::IRBuilder<>>(loop_block);
  // Exit
  llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(dynamic_counted_for->GetName(), "_exit_block"),
      node_context.llvm_function());
  std::unique_ptr<llvm::IRBuilder<>> exit_builder =
      std::make_unique<llvm::IRBuilder<>>(exit_block);

  // Get initial index, loop carry.
  llvm::Type* index_type = loop_body_function_type->getFunctionParamType(0);
  llvm::Value* init_index = llvm::ConstantInt::get(index_type, 0);
  llvm::Value* init_loop_carry = initial_value;

  // In entry, grab trip_count and stride, extended to match index type.
  // trip_count is zero-extended because the input trip_count is treated as
  // to be unsigned while stride is treated as signed.
  llvm::Value* trip_count_ext = b.CreateZExt(trip_count, index_type);
  llvm::Value* stride_ext = type_converter()->AsSignedValue(
      stride, dynamic_counted_for->stride()->GetType(), b, index_type);

  // Calculate index limit and jump entry loop predheader.
  llvm::Value* index_limit = b.CreateMul(trip_count_ext, stride_ext);
  b.CreateBr(preheader_block);

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
  llvm::Value* inc_index = loop_builder->CreateAdd(index_phi, stride_ext);
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

  return FinalizeNodeIrContext(node_context, args[1], std::move(exit_builder));
}

absl::Status IrBuilderVisitor::HandleEncode(Encode* encode) {
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(encode, {"operand"}));
  llvm::IRBuilder<>& b = node_context.builder();
  llvm::Value* input = node_context.operand(0);
  llvm::Type* input_type = input->getType();
  llvm::Value* input_one = llvm::ConstantInt::get(input_type, 1);

  llvm::Type* result_type =
      type_converter_->ConvertToLlvmType(encode->GetType());
  llvm::Value* result = llvm::ConstantInt::get(result_type, 0);

  llvm::Value* result_zero = llvm::ConstantInt::get(result_type, 0);

  // For each bit in the input, if it's set, bitwise-OR its [numeric] value
  // with the result.
  for (int i = 0; i < input_type->getIntegerBitWidth(); ++i) {
    llvm::Value* bit_set =
        b.CreateICmpEQ(b.CreateAnd(input, input_one), input_one);

    // Chained select, i.e., a = (b ? c : (d ? e : (...))), etc.
    llvm::Value* or_value = b.CreateSelect(
        bit_set, llvm::ConstantInt::get(result_type, i), result_zero);
    result = b.CreateOr(result, or_value);

    input = b.CreateLShr(input, input_one);
  }

  return FinalizeNodeIrContext(node_context, result);
}

absl::Status IrBuilderVisitor::HandleEq(CompareOp* eq) {
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(eq, {"lhs", "rhs"}));
  llvm::IRBuilder<>& b = node_context.builder();
  llvm::Value* llvm_lhs = node_context.operand(0);
  llvm::Value* llvm_rhs = node_context.operand(1);

  Node* lhs = eq->operand(0);
  Node* rhs = eq->operand(1);

  XLS_ASSIGN_OR_RETURN(std::vector<CompareTerm> eq_terms,
                       ExpandTerms(lhs, llvm_lhs, rhs, llvm_rhs, eq, &b));

  llvm::Value* result = b.getTrue();

  for (const auto& eq_term : eq_terms) {
    llvm::Value* term_test = b.CreateICmpEQ(eq_term.lhs, eq_term.rhs);
    result = b.CreateAnd(result, term_test);
  }

  return FinalizeNodeIrContext(node_context, result);
}

absl::Status IrBuilderVisitor::HandleGate(Gate* gate) {
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(gate, {"condition", "data"}));
  llvm::IRBuilder<>& b = node_context.builder();
  llvm::Value* condition = node_context.operand(0);
  llvm::Value* data = node_context.operand(1);

  XLS_ASSIGN_OR_RETURN(llvm::Constant * zero,
                       type_converter_->ToLlvmConstant(
                           gate->GetType(), ZeroOfType(gate->GetType())));
  llvm::Value* result = b.CreateSelect(condition, data, zero);

  return FinalizeNodeIrContext(node_context, result);
}

absl::Status IrBuilderVisitor::HandleIdentity(UnOp* identity) {
  return HandleUnaryOp(identity, [](llvm::Value* operand,
                                    llvm::IRBuilder<>& b) { return operand; });
}

absl::Status IrBuilderVisitor::HandleInvoke(Invoke* invoke) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(invoke, NumberedStrings("arg", invoke->operand_count()),
                       /*include_environment=*/true));
  llvm::IRBuilder<>& b = node_context.builder();

  XLS_ASSIGN_OR_RETURN(llvm::Function * function,
                       GetOrBuildFunction(invoke->to_apply()));

  std::vector<llvm::Value*> args(node_context.operands().begin(),
                                 node_context.operands().end());

  // Pass in the events, user_data, and runtime pointer to the loop body.
  args.push_back(node_context.GetInterpreterEvents());
  args.push_back(node_context.GetUserData());
  args.push_back(node_context.GetJitRuntime());

  llvm::Value* invoke_inst = b.CreateCall(function, args);
  return FinalizeNodeIrContext(node_context, invoke_inst);
}

absl::Status IrBuilderVisitor::HandleLiteral(Literal* literal) {
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(literal, {}));
  Type* xls_type = literal->GetType();
  XLS_ASSIGN_OR_RETURN(
      llvm::Value * llvm_literal,
      type_converter_->ToLlvmConstant(xls_type, literal->value()));
  return FinalizeNodeIrContext(node_context, llvm_literal);
}

absl::Status IrBuilderVisitor::HandleMap(Map* map) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(map, {"array"}, /*include_environment=*/true));
  llvm::IRBuilder<>& b = node_context.builder();

  XLS_ASSIGN_OR_RETURN(llvm::Function * to_apply,
                       GetOrBuildFunction(map->to_apply()));

  llvm::Value* input = node_context.operand(0);
  llvm::Type* input_type = input->getType();
  llvm::FunctionType* function_type = to_apply->getFunctionType();

  llvm::Value* result = CreateTypedZeroValue(llvm::ArrayType::get(
      function_type->getReturnType(), input_type->getArrayNumElements()));

  // Construct the arguments to pass to the map function.
  std::vector<llvm::Value*> args;
  args.push_back(nullptr);  // This is filled in each iteration of the loop.
  // Pass in the events, user_data, and runtime pointer to the loop body.
  args.push_back(node_context.GetInterpreterEvents());
  args.push_back(node_context.GetUserData());
  args.push_back(node_context.GetJitRuntime());

  for (uint32_t i = 0; i < input_type->getArrayNumElements(); ++i) {
    args[0] = b.CreateExtractValue(input, {i});
    llvm::Value* iter_result = b.CreateCall(to_apply, args);
    result = b.CreateInsertValue(result, iter_result, {i});
  }

  return FinalizeNodeIrContext(node_context, result);
}

absl::Status IrBuilderVisitor::HandleSMul(ArithOp* mul) {
  return HandleBinaryOpWithOperandConversion(
      mul,
      [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateMul(lhs, rhs);
      },
      /*is_signed=*/true);
}

absl::Status IrBuilderVisitor::HandleUMul(ArithOp* mul) {
  return HandleBinaryOpWithOperandConversion(
      mul,
      [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateMul(lhs, rhs);
      },
      /*is_signed=*/false);
}

absl::Status IrBuilderVisitor::HandleSMulp(PartialProductOp* mul) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(mul, NumberedStrings("operand", mul->operand_count())));

  llvm::IRBuilder<>& b = node_context.builder();

  llvm::Type* result_element_type = type_converter()->ConvertToLlvmType(
      mul->GetType()->AsTupleOrDie()->element_type(0));
  llvm::Type* tuple_type = type_converter_->ConvertToLlvmType(mul->GetType());
  llvm::Value* result = CreateTypedZeroValue(tuple_type);

  llvm::Value* lhs = type_converter()->AsSignedValue(node_context.operand(0),
                                                     mul->operand(0)->GetType(),
                                                     node_context.builder());
  llvm::Value* rhs = type_converter()->AsSignedValue(node_context.operand(1),
                                                     mul->operand(1)->GetType(),
                                                     node_context.builder());

  result = b.CreateInsertValue(
      result,
      b.CreateMul(b.CreateIntCast(lhs, result_element_type, /*isSigned=*/true),
                  b.CreateIntCast(rhs, result_element_type, /*isSigned=*/true)),
      {1});

  return FinalizeNodeIrContext(node_context, result);
}

absl::Status IrBuilderVisitor::HandleUMulp(PartialProductOp* mul) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(mul, NumberedStrings("operand", mul->operand_count())));

  llvm::IRBuilder<>& b = node_context.builder();

  llvm::Type* result_element_type = type_converter()->ConvertToLlvmType(
      mul->GetType()->AsTupleOrDie()->element_type(0));
  llvm::Type* tuple_type = type_converter_->ConvertToLlvmType(mul->GetType());
  llvm::Value* result = CreateTypedZeroValue(tuple_type);

  result = b.CreateInsertValue(
      result,
      b.CreateMul(b.CreateIntCast(node_context.operand(0), result_element_type,
                                  /*isSigned=*/false),
                  b.CreateIntCast(node_context.operand(1), result_element_type,
                                  /*isSigned=*/false)),
      {1});

  return FinalizeNodeIrContext(node_context, result);
}

absl::Status IrBuilderVisitor::HandleNaryAnd(NaryOp* and_op) {
  return HandleNaryOp(and_op, [](absl::Span<llvm::Value* const> operands,
                                 llvm::IRBuilder<>& b) {
    llvm::Value* result = operands.front();
    for (int i = 1; i < operands.size(); ++i) {
      result = b.CreateAnd(result, operands[i]);
    }
    return result;
  });
}

absl::Status IrBuilderVisitor::HandleNaryNand(NaryOp* nand_op) {
  return HandleNaryOp(nand_op, [](absl::Span<llvm::Value* const> operands,
                                  llvm::IRBuilder<>& b) {
    llvm::Value* result = operands.front();
    for (int i = 1; i < operands.size(); ++i) {
      result = b.CreateAnd(result, operands[i]);
    }
    return b.CreateNot(result);
  });
}

absl::Status IrBuilderVisitor::HandleNaryNor(NaryOp* nor_op) {
  return HandleNaryOp(nor_op, [](absl::Span<llvm::Value* const> operands,
                                 llvm::IRBuilder<>& b) {
    llvm::Value* result = operands.front();
    for (int i = 1; i < operands.size(); ++i) {
      result = b.CreateOr(result, operands[i]);
    }
    return b.CreateNot(result);
  });
}

absl::Status IrBuilderVisitor::HandleNaryOr(NaryOp* or_op) {
  return HandleNaryOp(
      or_op, [](absl::Span<llvm::Value* const> operands, llvm::IRBuilder<>& b) {
        llvm::Value* result = operands.front();
        for (int i = 1; i < operands.size(); ++i) {
          result = b.CreateOr(result, operands[i]);
        }
        return result;
      });
}

absl::Status IrBuilderVisitor::HandleNaryXor(NaryOp* xor_op) {
  return HandleNaryOp(xor_op, [](absl::Span<llvm::Value* const> operands,
                                 llvm::IRBuilder<>& b) {
    llvm::Value* result = operands.front();
    for (int i = 1; i < operands.size(); ++i) {
      result = b.CreateXor(result, operands[i]);
    }
    return result;
  });
}

absl::Status IrBuilderVisitor::HandleNe(CompareOp* ne) {
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(ne, {"lhs", "rhs"}));
  llvm::IRBuilder<>& b = node_context.builder();
  llvm::Value* llvm_lhs = node_context.operand(0);
  llvm::Value* llvm_rhs = node_context.operand(1);

  Node* lhs = ne->operand(0);
  Node* rhs = ne->operand(1);

  XLS_ASSIGN_OR_RETURN(std::vector<CompareTerm> ne_terms,
                       ExpandTerms(lhs, llvm_lhs, rhs, llvm_rhs, ne, &b));

  llvm::Value* result = b.getFalse();

  for (const auto& ne_term : ne_terms) {
    llvm::Value* term_test = b.CreateICmpNE(ne_term.lhs, ne_term.rhs);
    result = b.CreateOr(result, term_test);
  }

  return FinalizeNodeIrContext(node_context, result);
}

absl::Status IrBuilderVisitor::HandleNeg(UnOp* neg) {
  return HandleUnaryOp(neg, [](llvm::Value* operand, llvm::IRBuilder<>& b) {
    return b.CreateNeg(operand);
  });
}

absl::Status IrBuilderVisitor::HandleNot(UnOp* not_op) {
  return HandleUnaryOp(not_op, [](llvm::Value* operand, llvm::IRBuilder<>& b) {
    return b.CreateNot(operand);
  });
}

absl::Status IrBuilderVisitor::HandleOneHot(OneHot* one_hot) {
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(one_hot, {"operand"}));
  llvm::IRBuilder<>& b = node_context.builder();
  llvm::Value* input = node_context.operand(0);

  llvm::Type* input_type = input->getType();
  int input_width = input_type->getIntegerBitWidth();

  llvm::Value* llvm_false = b.getFalse();
  llvm::Value* llvm_true = b.getTrue();

  llvm::Value* result;
  if (one_hot->operand(0)->GetType()->AsBitsOrDie()->bit_count() > 0) {
    llvm::Value* zeroes;
    if (one_hot->priority() == LsbOrMsb::kLsb) {
      llvm::Function* cttz = llvm::Intrinsic::getDeclaration(
          module(), llvm::Intrinsic::cttz, {input_type});
      // We don't need to pass user data to these intrinsics; they're leaf
      // nodes.
      zeroes = b.CreateCall(cttz, {input, llvm_false});
    } else {
      llvm::Function* ctlz = llvm::Intrinsic::getDeclaration(
          module(), llvm::Intrinsic::ctlz, {input_type});
      zeroes = b.CreateCall(ctlz, {input, llvm_false});
      zeroes = b.CreateSub(llvm::ConstantInt::get(input_type, input_width - 1),
                           zeroes);
    }

    // If the input is zero, then return the special high-bit value.
    llvm::Value* zero_value = llvm::ConstantInt::get(input_type, 0);
    llvm::Value* width_value = llvm::ConstantInt::get(
        input_type, one_hot->operand(0)->GetType()->GetFlatBitCount());
    llvm::Value* eq_zero = b.CreateICmpEQ(input, zero_value);
    llvm::Value* shift_amount = b.CreateSelect(eq_zero, width_value, zeroes);

    llvm::Type* result_type =
        type_converter()->ConvertToLlvmType(one_hot->GetType());
    result = b.CreateShl(llvm::ConstantInt::get(result_type, 1),
                         b.CreateZExt(shift_amount, result_type));
  } else {
    result = llvm_true;
  }

  return FinalizeNodeIrContext(node_context, result);
}

absl::Status IrBuilderVisitor::HandleOneHotSel(OneHotSelect* sel) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(
          sel, ConcatVectors({"selector"},
                             NumberedStrings("case", sel->cases().size()))));
  llvm::IRBuilder<>& b = node_context.builder();
  llvm::Value* selector = node_context.operand(0);
  absl::Span<llvm::Value* const> cases = node_context.operands().subspan(1);
  llvm::Type* input_type = cases.front()->getType();

  llvm::Value* result;
  result = CreateTypedZeroValue(input_type);

  llvm::Value* typed_zero = CreateTypedZeroValue(input_type);
  llvm::Value* llvm_one = llvm::ConstantInt::get(selector->getType(), 1);

  for (llvm::Value* cse : cases) {
    // Extract the current selector bit & see if set (CreateSelect requires an
    // i1 argument, or we could directly use the AND result.
    llvm::Value* is_hot =
        b.CreateICmpEQ(b.CreateAnd(selector, llvm_one), llvm_one);

    // OR with zero might be slower than doing an if/else construct - if
    // it turns out to be performance-critical, we can update it.
    llvm::Value* or_value = b.CreateSelect(is_hot, cse, typed_zero);
    result = CreateAggregateOr(result, or_value, &b);
    selector = b.CreateLShr(selector, llvm_one);
  }

  return FinalizeNodeIrContext(node_context, result);
}

absl::Status IrBuilderVisitor::HandlePrioritySel(PrioritySelect* sel) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(
          sel, ConcatVectors({"selector"},
                             NumberedStrings("case", sel->cases().size()))));
  llvm::IRBuilder<>& b = node_context.builder();
  llvm::Value* selector = node_context.operand(0);
  absl::Span<llvm::Value* const> cases = node_context.operands().subspan(1);
  llvm::Type* input_type = cases.front()->getType();

  llvm::Value* typed_zero = CreateTypedZeroValue(input_type);
  llvm::Value* llvm_false = b.getFalse();

  // Get index to select by counting trailing zeros
  llvm::Function* cttz = llvm::Intrinsic::getDeclaration(
      module(), llvm::Intrinsic::cttz, {selector->getType()});
  llvm::Value* selected_index = b.CreateCall(cttz, {selector, llvm_false});

  // Sel is implemented by a cascading series of select ops, e.g.,
  // selector == 0 ? cases[0] : selector == 1 ? cases[1] : selector == 2 ? ...
  llvm::Value* llvm_sel = typed_zero;
  for (int i = sel->cases().size() - 1; i >= 0; i--) {
    llvm::Value* current_index = llvm::ConstantInt::get(selector->getType(), i);
    llvm::Value* cmp = b.CreateICmpEQ(selected_index, current_index);
    llvm_sel = b.CreateSelect(cmp, cases.at(i), llvm_sel);
  }

  llvm::Value* selector_is_zero =
      b.CreateICmpEQ(selector, llvm::ConstantInt::get(selector->getType(), 0));
  llvm_sel = b.CreateSelect(selector_is_zero, typed_zero, llvm_sel);
  return FinalizeNodeIrContext(node_context, llvm_sel);
}

absl::Status IrBuilderVisitor::HandleOrReduce(BitwiseReductionOp* op) {
  return HandleUnaryOp(op, [](llvm::Value* operand, llvm::IRBuilder<>& b) {
    // OR-reduce is equivalent to checking if any bit is set in the input.
    return b.CreateICmpNE(operand,
                          llvm::ConstantInt::get(operand->getType(), 0));
  });
}

absl::Status IrBuilderVisitor::HandleReverse(UnOp* reverse) {
  return HandleUnaryOp(
      reverse, [&](llvm::Value* operand, llvm::IRBuilder<>& b) {
        llvm::Function* reverse_fn = llvm::Intrinsic::getDeclaration(
            module(), llvm::Intrinsic::bitreverse, {operand->getType()});
        // Shift right logically by native width - natural with.
        llvm::Value* result = b.CreateCall(reverse_fn, {operand});
        result = b.CreateLShr(
            result,
            llvm::ConstantInt::get(result->getType(),
                                   type_converter()->GetLlvmBitCount(
                                       reverse->GetType()->AsBitsOrDie()) -
                                       reverse->GetType()->GetFlatBitCount()));
        return result;
      });
}

absl::Status IrBuilderVisitor::HandleSDiv(BinOp* binop) {
  return HandleBinaryOp(
      binop,
      [&](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return EmitDiv(lhs, rhs, /*is_signed=*/true, type_converter(), &b);
      },
      /*is_signed=*/true);
}

absl::Status IrBuilderVisitor::HandleSMod(BinOp* binop) {
  return HandleBinaryOp(
      binop,
      [&](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return EmitMod(lhs, rhs, /*is_signed=*/true, &b);
      },
      /*is_signed=*/true);
}

absl::Status IrBuilderVisitor::HandleSel(Select* sel) {
  std::vector<std::string> operand_names =
      ConcatVectors({"selector"}, NumberedStrings("case", sel->cases().size()));
  if (sel->default_value().has_value()) {
    operand_names.push_back("default");
  }
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(sel, operand_names));

  llvm::IRBuilder<>& b = node_context.builder();

  // Sel is implemented by a cascading series of select ops, e.g.,
  // selector == 0 ? cases[0] : selector == 1 ? cases[1] : selector == 2 ? ...
  llvm::Value* selector = node_context.operand(0);
  llvm::Value* llvm_sel =
      sel->default_value() ? node_context.operands().back() : nullptr;
  for (int i = sel->cases().size() - 1; i >= 0; i--) {
    llvm::Value* llvm_case = node_context.operand(i + 1);
    if (llvm_sel == nullptr) {
      // The last element in the select tree isn't a sel, but an actual value.
      llvm_sel = llvm_case;
    } else {
      llvm::Value* index = llvm::ConstantInt::get(selector->getType(), i);
      llvm::Value* cmp = b.CreateICmpEQ(selector, index);
      llvm_sel = b.CreateSelect(cmp, llvm_case, llvm_sel);
    }
  }
  return FinalizeNodeIrContext(node_context, llvm_sel);
}

absl::Status IrBuilderVisitor::HandleSGe(CompareOp* ge) {
  return HandleBinaryOp(
      ge,
      [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateICmpSGE(lhs, rhs);
      },
      /*is_signed=*/true);
}

absl::Status IrBuilderVisitor::HandleSGt(CompareOp* gt) {
  return HandleBinaryOp(
      gt,
      [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateICmpSGT(lhs, rhs);
      },
      /*is_signed=*/true);
}

absl::Status IrBuilderVisitor::HandleSignExtend(ExtendOp* sign_ext) {
  return HandleUnaryOp(
      sign_ext,
      [&](llvm::Value* operand, llvm::IRBuilder<>& b) {
        llvm::Type* new_type =
            type_converter_->ConvertToLlvmType(sign_ext->GetType());
        return b.CreateSExt(operand, new_type);
      },
      /*is_signed=*/true);
}

absl::Status IrBuilderVisitor::HandleSLe(CompareOp* le) {
  return HandleBinaryOp(
      le,
      [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateICmpSLE(lhs, rhs);
      },
      /*is_signed=*/true);
}

absl::Status IrBuilderVisitor::HandleSLt(CompareOp* lt) {
  return HandleBinaryOp(
      lt,
      [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateICmpSLT(lhs, rhs);
      },
      /*is_signed=*/true);
}

absl::Status IrBuilderVisitor::HandleShll(BinOp* binop) {
  return HandleBinaryOp(
      binop, [&](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return EmitShiftOp(binop, lhs, rhs, &b, type_converter());
      });
}

absl::Status IrBuilderVisitor::HandleShra(BinOp* binop) {
  return HandleBinaryOp(
      binop, [&](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        // Only the LHS is treated as a signed number.
        return EmitShiftOp(binop,
                           type_converter()->AsSignedValue(
                               lhs, binop->operand(0)->GetType(), b),
                           rhs, &b, type_converter());
      });
}

absl::Status IrBuilderVisitor::HandleShrl(BinOp* binop) {
  return HandleBinaryOp(
      binop, [&](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return EmitShiftOp(binop, lhs, rhs, &b, type_converter());
      });
}

absl::Status IrBuilderVisitor::HandleSub(BinOp* binop) {
  return HandleBinaryOp(
      binop, [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateSub(lhs, rhs);
      });
}

absl::Status IrBuilderVisitor::HandleTuple(Tuple* tuple) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(tuple,
                       NumberedStrings("operand", tuple->operand_count())));
  llvm::IRBuilder<>& b = node_context.builder();

  llvm::Type* tuple_type = type_converter_->ConvertToLlvmType(tuple->GetType());

  llvm::Value* result = CreateTypedZeroValue(tuple_type);
  for (uint32_t i = 0; i < tuple->operand_count(); ++i) {
    if (tuple->operand(i)->GetType()->GetFlatBitCount() == 0) {
      continue;
    }
    result = b.CreateInsertValue(result, node_context.operand(i), {i});
  }

  return FinalizeNodeIrContext(node_context, result);
}

absl::Status IrBuilderVisitor::HandleTupleIndex(TupleIndex* index) {
  return HandleUnaryOp(index, [&](llvm::Value* operand, llvm::IRBuilder<>& b) {
    return b.CreateExtractValue(operand, index->index());
  });
}

absl::Status IrBuilderVisitor::HandleUDiv(BinOp* binop) {
  return HandleBinaryOp(
      binop, [&](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return EmitDiv(lhs, rhs, /*is_signed=*/false, type_converter(), &b);
      });
}

absl::Status IrBuilderVisitor::HandleUMod(BinOp* binop) {
  return HandleBinaryOp(
      binop, [&](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return EmitMod(lhs, rhs, /*is_signed=*/false, &b);
      });
}

absl::Status IrBuilderVisitor::HandleUGe(CompareOp* ge) {
  return HandleBinaryOp(
      ge, [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateICmpUGE(lhs, rhs);
      });
}

absl::Status IrBuilderVisitor::HandleUGt(CompareOp* gt) {
  return HandleBinaryOp(
      gt, [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateICmpUGT(lhs, rhs);
      });
}

absl::Status IrBuilderVisitor::HandleULe(CompareOp* le) {
  return HandleBinaryOp(
      le, [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateICmpULE(lhs, rhs);
      });
}

absl::Status IrBuilderVisitor::HandleULt(CompareOp* lt) {
  return HandleBinaryOp(
      lt, [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateICmpULT(lhs, rhs);
      });
}

absl::Status IrBuilderVisitor::HandleXorReduce(BitwiseReductionOp* op) {
  return HandleUnaryOp(op, [&](llvm::Value* operand, llvm::IRBuilder<>& b) {
    // XOR-reduce is equivalent to checking if the number of set bits is odd.
    llvm::Function* ctpop = llvm::Intrinsic::getDeclaration(
        module(), llvm::Intrinsic::ctpop, {operand->getType()});
    // We don't need to pass user data to intrinsics; they're leaf nodes.
    llvm::Value* pop_count = b.CreateCall(ctpop, {operand});

    // Once we have the pop count, truncate to the first (i.e., "is odd") bit.
    return b.CreateTrunc(pop_count, llvm::IntegerType::get(ctx(), 1));
  });
}

absl::Status IrBuilderVisitor::HandleZeroExtend(ExtendOp* zero_ext) {
  return HandleUnaryOp(
      zero_ext, [&](llvm::Value* operand, llvm::IRBuilder<>& b) {
        llvm::Type* new_type =
            type_converter_->ConvertToLlvmType(zero_ext->GetType());
        return b.CreateZExt(operand, new_type);
      });
}

absl::Status IrBuilderVisitor::HandleUnaryOp(
    Node* node,
    std::function<llvm::Value*(llvm::Value*, llvm::IRBuilder<>&)> build_result,
    bool is_signed) {
  XLS_RET_CHECK_EQ(node->operand_count(), 1);
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(node, {"operand"}));
  return FinalizeNodeIrContext(
      node_context,
      build_result(
          MaybeAsSigned(node_context.operand(0), node->operand(0)->GetType(),
                        node_context.builder(), is_signed),
          node_context.builder()));
}

absl::Status IrBuilderVisitor::HandleBinaryOp(
    Node* node,
    std::function<llvm::Value*(llvm::Value*, llvm::Value*, llvm::IRBuilder<>&)>
        build_result,
    bool is_signed) {
  XLS_RET_CHECK_EQ(node->operand_count(), 2);
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(node, {"lhs", "rhs"}));
  llvm::Value* lhs =
      MaybeAsSigned(node_context.operand(0), node->operand(0)->GetType(),
                    node_context.builder(), is_signed);
  llvm::Value* rhs =
      MaybeAsSigned(node_context.operand(1), node->operand(1)->GetType(),
                    node_context.builder(), is_signed);
  return FinalizeNodeIrContext(node_context,
                               build_result(lhs, rhs, node_context.builder()));
}

absl::Status IrBuilderVisitor::HandleNaryOp(
    Node* node, std::function<llvm::Value*(absl::Span<llvm::Value* const>,
                                           llvm::IRBuilder<>&)>
                    build_result) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(node,
                       NumberedStrings("operand", node->operand_count())));
  return FinalizeNodeIrContext(
      node_context,
      build_result(node_context.operands(), node_context.builder()));
}

absl::Status IrBuilderVisitor::HandleBinaryOpWithOperandConversion(
    Node* node,
    std::function<llvm::Value*(llvm::Value*, llvm::Value*, llvm::IRBuilder<>&)>
        build_result,
    bool is_signed) {
  XLS_RET_CHECK_EQ(node->operand_count(), 2);
  llvm::Type* result_type = type_converter_->ConvertToLlvmType(node->GetType());
  return HandleBinaryOp(
      node, [&](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        llvm::Value* converted_lhs =
            MaybeAsSigned(lhs, node->operand(0)->GetType(), b, is_signed);
        llvm::Value* converted_rhs =
            MaybeAsSigned(rhs, node->operand(1)->GetType(), b, is_signed);
        return build_result(
            b.CreateIntCast(converted_lhs, result_type, is_signed),
            b.CreateIntCast(converted_rhs, result_type, is_signed), b);
      });
}

llvm::Constant* IrBuilderVisitor::CreateTypedZeroValue(llvm::Type* type) {
  if (type->isIntegerTy()) {
    return llvm::ConstantInt::get(type, 0);
  }
  if (type->isArrayTy()) {
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

llvm::Value* IrBuilderVisitor::LoadFromPointerArray(
    int64_t index, llvm::Type* data_type, llvm::Value* pointer_array,
    int64_t pointer_array_size, llvm::IRBuilder<>* builder) {
  llvm::Value* data_ptr = LoadPointerFromPointerArray(
      index, data_type, pointer_array, pointer_array_size, builder);
  return builder->CreateLoad(data_type, data_ptr);
}

llvm::Value* IrBuilderVisitor::LoadPointerFromPointerArray(
    int64_t index, llvm::Type* data_type, llvm::Value* pointer_array,
    int64_t pointer_array_size, llvm::IRBuilder<>* builder) {
  llvm::LLVMContext& context = builder->getContext();
  llvm::Type* pointer_array_type = llvm::ArrayType::get(
      llvm::Type::getInt8PtrTy(context), pointer_array_size);
  llvm::Value* gep = builder->CreateGEP(
      pointer_array_type, pointer_array,
      {
          llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 0),
          llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), index),
      });

  return builder->CreateLoad(llvm::PointerType::get(context, 0), gep);
}

absl::Status IrBuilderVisitor::StoreResult(Node* node, llvm::Value* value) {
  XLS_RET_CHECK(!node_map_.contains(node));
  value->setName(verilog::SanitizeIdentifier(node->GetName()));
  node_map_[node] = value;

  return absl::OkStatus();
}

void IrBuilderVisitor::UnpoisonBuffer(llvm::Value* buffer, int64_t size,
                                      llvm::IRBuilder<>* builder) {
#ifdef ABSL_HAVE_MEMORY_SANITIZER
  llvm::LLVMContext& context = builder->getContext();
  llvm::ConstantInt* fn_addr =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(context),
                             absl::bit_cast<uint64_t>(&__msan_unpoison));
  llvm::Type* void_type = llvm::Type::getVoidTy(context);
  llvm::Type* ptr_type = llvm::PointerType::get(builder->getContext(), 0);
  llvm::Type* size_t_type =
      llvm::Type::getIntNTy(context, sizeof(size_t) * CHAR_BIT);
  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(void_type, {ptr_type, size_t_type}, false);
  llvm::Value* fn_ptr =
      builder->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));

  std::vector<llvm::Value*> args = {buffer,
                                    llvm::ConstantInt::get(size_t_type, size)};

  builder->CreateCall(fn_type, fn_ptr, args);
#endif
}

absl::StatusOr<llvm::Function*> IrBuilderVisitor::GetOrBuildFunction(
    Function* function) {
  // If we've not processed this function yet, then do so.
  llvm::Function* found_function =
      module()->getFunction(function->qualified_name());
  if (found_function != nullptr) {
    return found_function;
  }

  // Build the LLVM function for this XLS function and return it.
  return function_builder_(function);
}

absl::StatusOr<NodeIrContext> IrBuilderVisitor::NewNodeIrContext(
    Node* node, absl::Span<const std::string> operand_names,
    bool include_environment) {
  std::optional<NodeIrContext::Environment> environment;
  if (include_environment) {
    environment = {GetInterpreterEventsPtr(), GetUserDataPtr(),
                   GetJitRuntimePtr()};
  }
  return NodeIrContext::Create(node, operand_names, type_converter(), module(),
                               environment);
}

absl::Status IrBuilderVisitor::FinalizeNodeIrContext(
    NodeIrContext& node_context, llvm::Value* result,
    std::optional<std::unique_ptr<llvm::IRBuilder<>>> exit_builder) {
  node_context.Finalize(
      result, exit_builder.has_value()
                  ? std::optional<llvm::IRBuilder<>*>(exit_builder->get())
                  : std::nullopt);

  std::vector<llvm::Value*> args;
  for (Node* operand : node_context.node()->operands()) {
    args.push_back(node_map_.at(operand));
  }
  if (node_context.HasEnvironment()) {
    args.push_back(GetInterpreterEventsPtr());
    args.push_back(GetUserDataPtr());
    args.push_back(GetJitRuntimePtr());
  }

  return StoreResult(
      node_context.node(),
      dispatch_builder()->CreateCall(node_context.llvm_function(), args));
}

}  // namespace xls
