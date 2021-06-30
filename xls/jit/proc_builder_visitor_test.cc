// Copyright 2021 The XLS Authors
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

#include "xls/jit/proc_builder_visitor.h"

#include <filesystem>
#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "llvm/include/llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/include/llvm/ExecutionEngine/GenericValue.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/include/llvm/IR/Function.h"
#include "llvm/include/llvm/IR/Module.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/subprocess.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/llvm_type_converter.h"

namespace xls {
namespace {

// One-time init for LLVM.
absl::once_flag once;
void OnceInit() {
  LLVMInitializeNativeTarget();
  LLVMInitializeNativeAsmPrinter();
  LLVMInitializeNativeAsmParser();
}

void EnqueueData(JitChannelQueue* queue, uint32_t data) {
  queue->Send(absl::bit_cast<uint8_t*>(&data), sizeof(uint32_t));
}

uint32_t DequeueData(JitChannelQueue* queue) {
  uint32_t data;
  queue->Recv(absl::bit_cast<uint8_t*>(&data), sizeof(uint32_t));
  return data;
}

// There's a lot of boilerplate necessary to interpret our generated IR!
// Thus, we'll use a fixture to wrap most of it up.
class ProcBuilderVisitorTest : public ::testing::Test {
 protected:
  using EntryFunctionT = void (*)(uint8_t**, uint8_t*, void*, void*);
  static constexpr const char kModuleName[] = "the_module";
  static constexpr const char kFunctionName[] = "the_function";

  virtual void SetUp() { absl::call_once(once, OnceInit); }

  absl::Status InitPackage(const std::string& ir_text) {
    XLS_ASSIGN_OR_RETURN(package_, Parser::ParsePackage(ir_text));
    return absl::OkStatus();
  }

  // "num_params" is specifiable for non-Proc functions. Procs will always have
  // two params - a token and a state element.
  absl::Status InitLlvm(llvm::Module* module, Type* return_type,
                        int num_params = 2) {
    auto target_builder = llvm::orc::JITTargetMachineBuilder::detectHost();
    XLS_RET_CHECK(target_builder) << llvm::toString(target_builder.takeError());

    auto target_machine = target_builder->createTargetMachine();
    XLS_RET_CHECK(target_machine) << llvm::toString(target_machine.takeError());
    data_layout_ = std::make_unique<llvm::DataLayout>(
        target_machine.get()->createDataLayout());

    std::vector<llvm::Type*> llvm_param_types;
    llvm_param_types.push_back(llvm::PointerType::get(
        llvm::ArrayType::get(llvm::Type::getInt8PtrTy(context_, /*AS=*/0),
                             num_params),
        /*AddressSpace=*/0));
    type_converter_ =
        std::make_unique<LlvmTypeConverter>(&context_, *data_layout_);

    llvm::Type* llvm_return_type =
        type_converter_->ConvertToLlvmType(return_type);
    llvm_param_types.push_back(
        llvm::PointerType::get(llvm_return_type, /*AddressSpace=*/0));

    // The assert status pointer needs an argument too.
    llvm_param_types.push_back(llvm::Type::getInt64Ty(context_));

    // Don't forget the user data pointer! Because I did the first time!
    llvm_param_types.push_back(llvm::Type::getInt64Ty(context_));

    llvm::FunctionType* fn_type = llvm::FunctionType::get(
        llvm::Type::getVoidTy(context_), llvm_param_types, /*isVarArg=*/false);
    llvm_fn_ = llvm::cast<llvm::Function>(
        module->getOrInsertFunction(kFunctionName, fn_type).getCallee());
    return absl::OkStatus();
  }

  // Caution! This invalidates module_!
  EntryFunctionT BuildEntryFn(std::unique_ptr<llvm::Module> module,
                              const std::string& fn_name) {
    llvm::EngineBuilder builder(std::move(module));
    builder.setEngineKind(llvm::EngineKind::JIT);
    evaluator_ = absl::WrapUnique(builder.create());
    return absl::bit_cast<EntryFunctionT>(
        evaluator_->getFunctionAddress(fn_name));
  }

  // Packs the input values into the format our LLVM functions expect (i.e.,
  // input params as an array of u8 pointers). The resulting pointers are held
  // as unique_ptrs to auto-magically handle dealloc.
  absl::StatusOr<std::vector<uint8_t*>> PackArgs(
      FunctionBase* function, absl::Span<const Value> values) {
    std::vector<Type*> param_types;
    for (Param* param : function->params()) {
      param_types.push_back(param->GetType());
    }
    XLS_RET_CHECK(values.size() == param_types.size());

    JitRuntime runtime(*data_layout(), type_converter());
    std::vector<uint8_t*> arg_buffers;
    unique_arg_buffers_.clear();
    unique_arg_buffers_.reserve(param_types.size());
    arg_buffers.reserve(param_types.size());

    for (const Type* type : param_types) {
      unique_arg_buffers_.push_back(
          std::make_unique<uint8_t[]>(type_converter()->GetTypeByteSize(type)));
      arg_buffers.push_back(unique_arg_buffers_.back().get());
    }

    XLS_RETURN_IF_ERROR(
        runtime.PackArgs(values, param_types, absl::MakeSpan(arg_buffers)));
    return arg_buffers;
  }

  Package* package() { return package_.get(); }
  llvm::LLVMContext& ctx() { return context_; }
  llvm::Function* llvm_fn() { return llvm_fn_; }
  llvm::DataLayout* data_layout() { return data_layout_.get(); }
  LlvmTypeConverter* type_converter() { return type_converter_.get(); }

 private:
  llvm::LLVMContext context_;
  llvm::Function* llvm_fn_;
  std::unique_ptr<Package> package_;

  std::unique_ptr<llvm::DataLayout> data_layout_;
  std::unique_ptr<LlvmTypeConverter> type_converter_;
  std::unique_ptr<llvm::ExecutionEngine> evaluator_;

  std::vector<std::unique_ptr<uint8_t[]>> unique_arg_buffers_;
};

// Simple smoke-style test verifying that ProcBuilderVisitor can still build
// regular functions.
TEST_F(ProcBuilderVisitorTest, CanCompileFunctions) {
  const std::string kIrText = R"(
package p

fn AddTwo(a: bits[8], b: bits[8]) -> bits[8] {
  ret add.1: bits[8] = add(a, b)
}
)";

  XLS_ASSERT_OK(InitPackage(kIrText));
  Type* u8_type = package()->GetBitsType(8);

  auto module = std::make_unique<llvm::Module>(kModuleName, ctx());
  XLS_ASSERT_OK(InitLlvm(module.get(), u8_type, /*num_params=*/2));
  XLS_ASSERT_OK_AND_ASSIGN(auto xls_fn, package()->GetFunction("AddTwo"));

  XLS_ASSERT_OK_AND_ASSIGN(auto queue_mgr,
                           JitChannelQueueManager::Create(package()));
  XLS_ASSERT_OK(ProcBuilderVisitor::Visit(
      module.get(), llvm_fn(), xls_fn, type_converter(), /*is_top=*/true,
      /*generate_packed=*/false, queue_mgr.get(), nullptr, nullptr));
  JitRuntime runtime(*data_layout(), type_converter());

  // The Interpreter takes "generic values"; we need to pass a pointer into our
  // function - one for the arg array, and one for the return value.
  uint8_t* input_buffer[2];
  uint8_t arg_0 = 47;
  uint8_t arg_1 = 33;
  input_buffer[0] = &arg_0;
  input_buffer[1] = &arg_1;
  std::vector<llvm::GenericValue> args;
  args.push_back(llvm::GenericValue(input_buffer));
  uint8_t output_buffer;
  args.push_back(llvm::GenericValue(&output_buffer));

  llvm::EngineBuilder builder(std::move(module));
  builder.setEngineKind(llvm::EngineKind::JIT);
  std::unique_ptr<llvm::ExecutionEngine> evaluator =
      absl::WrapUnique(builder.create());
  ASSERT_TRUE(evaluator != nullptr);
  auto fn = absl::bit_cast<EntryFunctionT>(
      evaluator->getFunctionAddress(kFunctionName));
  fn(input_buffer, &output_buffer, nullptr, nullptr);
  EXPECT_EQ(output_buffer, arg_0 + arg_1);
}

// Recv/Send functions for the "CanCompileProcs" test.
void CanCompileProcs_recv(JitChannelQueue* queue_ptr, Receive* recv_ptr,
                          uint8_t* data_ptr, int64_t data_sz, void* user_data) {
  JitChannelQueue* queue = absl::bit_cast<JitChannelQueue*>(queue_ptr);
  queue->Recv(data_ptr, data_sz);
}

void CanCompileProcs_send(JitChannelQueue* queue_ptr, Send* send_ptr,
                          uint8_t* data_ptr, int64_t data_sz, void* user_data) {
  JitChannelQueue* queue = absl::bit_cast<JitChannelQueue*>(queue_ptr);
  queue->Send(data_ptr, data_sz);
}

// Simple smoke-style test that the ProcBuilderVisitor can compile Procs!
TEST_F(ProcBuilderVisitorTest, CanCompileProcs) {
  const std::string kIrText = R"(
package p

chan c_i(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=none, metadata="")
chan c_o(bits[32], id=1, kind=streaming, ops=send_only, flow_control=none, metadata="")

proc the_proc(my_token: token, state: (), init=()) {
  literal.1: bits[32] = literal(value=3)
  receive.2: (token, bits[32]) = receive(my_token, channel_id=0)
  tuple_index.3: token = tuple_index(receive.2, index=0)
  tuple_index.4: bits[32] = tuple_index(receive.2, index=1)
  umul.5: bits[32] = umul(literal.1, tuple_index.4)
  send.6: token = send(tuple_index.3, umul.5, channel_id=1)
  next (send.6, state)
}
)";
  XLS_ASSERT_OK(InitPackage(kIrText));
  auto module = std::make_unique<llvm::Module>(kModuleName, ctx());
  XLS_ASSERT_OK(InitLlvm(module.get(), package()->GetTupleType({})));
  XLS_ASSERT_OK_AND_ASSIGN(auto xls_fn, package()->GetProc("the_proc"));

  XLS_ASSERT_OK_AND_ASSIGN(auto queue_mgr,
                           JitChannelQueueManager::Create(package()));
  EnqueueData(queue_mgr->GetQueueById(0).value(), 7);

  XLS_ASSERT_OK(ProcBuilderVisitor::Visit(
      module.get(), llvm_fn(), xls_fn, type_converter(),
      /*is_top=*/true, /*generate_packed=*/false, queue_mgr.get(),
      &CanCompileProcs_recv, &CanCompileProcs_send));

  // The provided JIT doesn't support ExecutionEngine::runFunction, so we have
  // to get the fn pointer and call that directly.
  auto fn = BuildEntryFn(std::move(module), kFunctionName);

  // We don't have any persistent state, so we don't reference params, hence
  // nullptr.
  fn(nullptr, nullptr, nullptr, nullptr);
  EXPECT_EQ(DequeueData(queue_mgr->GetQueueById(1).value()), 21);

  // Let's make sure we can call it 2x!
  EnqueueData(queue_mgr->GetQueueById(0).value(), 8);
  fn(nullptr, nullptr, nullptr, nullptr);
  EXPECT_EQ(DequeueData(queue_mgr->GetQueueById(1).value()), 24);
}

TEST_F(ProcBuilderVisitorTest, RecvIf) {
  const std::string kIrText = R"(
package p

chan c_i(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=none, metadata="")
chan c_o(bits[32], id=1, kind=streaming, ops=send_only, flow_control=none, metadata="")

proc the_proc(my_token: token, state: bits[1], init=0) {
  receive.2: (token, bits[32]) = receive(my_token, predicate=state, channel_id=0)
  tuple_index.3: token = tuple_index(receive.2, index=0)
  tuple_index.4: bits[32] = tuple_index(receive.2, index=1)
  send.5: token = send(tuple_index.3, tuple_index.4, channel_id=1)
  next (send.5, state)
}
)";
  XLS_ASSERT_OK(InitPackage(kIrText));
  auto module = std::make_unique<llvm::Module>(kModuleName, ctx());
  XLS_ASSERT_OK(InitLlvm(module.get(), package()->GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(auto xls_fn, package()->GetProc("the_proc"));

  constexpr uint32_t kQueueData = 0xbeef;
  XLS_ASSERT_OK_AND_ASSIGN(auto queue_mgr,
                           JitChannelQueueManager::Create(package()));
  EnqueueData(queue_mgr->GetQueueById(0).value(), kQueueData);

  XLS_ASSERT_OK(ProcBuilderVisitor::Visit(
      module.get(), llvm_fn(), xls_fn, type_converter(),
      /*is_top=*/true, /*generate_packed=*/false, queue_mgr.get(),
      &CanCompileProcs_recv, &CanCompileProcs_send));

  // First: set state to 0; see that recv_if returns 0.
  uint64_t output;
  auto fn = BuildEntryFn(std::move(module), kFunctionName);
  std::vector<Value> args = {Value::Token(), Value(UBits(0, 1))};
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<uint8_t*> arg_buffers,
                           PackArgs(xls_fn, args));
  fn(arg_buffers.data(), absl::bit_cast<uint8_t*>(&output), nullptr, nullptr);
  EXPECT_EQ(DequeueData(queue_mgr->GetQueueById(1).value()), 0);

  // Second: set state to 1, see that recv_if returns what we put in the queue
  args = {Value::Token(), Value(UBits(1, 1))};
  XLS_ASSERT_OK_AND_ASSIGN(arg_buffers, PackArgs(xls_fn, args));
  fn(arg_buffers.data(), absl::bit_cast<uint8_t*>(&output), nullptr, nullptr);
  EXPECT_EQ(DequeueData(queue_mgr->GetQueueById(1).value()), kQueueData);
}

TEST_F(ProcBuilderVisitorTest, ConditionalSend) {
  const std::string kIrText = R"(
package p

chan c_i(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=none, metadata="")
chan c_o(bits[32], id=1, kind=streaming, ops=send_only, flow_control=none, metadata="")

proc the_proc(my_token: token, state: bits[1], init=0) {
  receive.2: (token, bits[32]) = receive(my_token, channel_id=0)
  tuple_index.3: token = tuple_index(receive.2, index=0)
  tuple_index.4: bits[32] = tuple_index(receive.2, index=1)
  send.5: token = send(tuple_index.3, tuple_index.4, predicate=state, channel_id=1)
  next (send.5, state)
}
)";
  XLS_ASSERT_OK(InitPackage(kIrText));
  auto module = std::make_unique<llvm::Module>(kModuleName, ctx());
  XLS_ASSERT_OK(InitLlvm(module.get(), package()->GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(auto xls_fn, package()->GetProc("the_proc"));

  constexpr uint32_t kQueueData = 0xbeef;
  XLS_ASSERT_OK_AND_ASSIGN(auto queue_mgr,
                           JitChannelQueueManager::Create(package()));
  EnqueueData(queue_mgr->GetQueueById(0).value(), kQueueData);
  EnqueueData(queue_mgr->GetQueueById(0).value(), kQueueData + 1);

  XLS_ASSERT_OK(ProcBuilderVisitor::Visit(
      module.get(), llvm_fn(), xls_fn, type_converter(),
      /*is_top=*/true, /*generate_packed=*/false, queue_mgr.get(),
      &CanCompileProcs_recv, &CanCompileProcs_send));

  // First: with state 0, make sure no send occurred (i.e., our output queue is
  // empty).
  uint64_t output;
  auto fn = BuildEntryFn(std::move(module), kFunctionName);
  std::vector<Value> args = {Value::Token(), Value(UBits(0, 1))};
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<uint8_t*> arg_buffers,
                           PackArgs(xls_fn, args));
  fn(arg_buffers.data(), absl::bit_cast<uint8_t*>(&output), nullptr, nullptr);

  // Second: with state 1, make sure we've now got output data.
  args = {Value::Token(), Value(UBits(1, 1))};
  XLS_ASSERT_OK_AND_ASSIGN(arg_buffers, PackArgs(xls_fn, args));
  fn(arg_buffers.data(), absl::bit_cast<uint8_t*>(&output), nullptr, nullptr);
  EXPECT_EQ(DequeueData(queue_mgr->GetQueueById(1).value()), kQueueData + 1);
}

// Recv/Send functions for the "GetsUserData" test.
void GetsUserData_recv(JitChannelQueue* queue_ptr, Receive* recv_ptr,
                       uint8_t* data_ptr, int64_t data_sz, void* user_data) {
  JitChannelQueue* queue = absl::bit_cast<JitChannelQueue*>(queue_ptr);
  uint64_t* int_data = absl::bit_cast<uint64_t*>(user_data);
  *int_data = *int_data * 2;
  queue->Recv(data_ptr, data_sz);
}

void GetsUserData_send(JitChannelQueue* queue_ptr, Send* send_ptr,
                       uint8_t* data_ptr, int64_t data_sz, void* user_data) {
  JitChannelQueue* queue = absl::bit_cast<JitChannelQueue*>(queue_ptr);
  uint64_t* int_data = absl::bit_cast<uint64_t*>(user_data);
  *int_data = *int_data * 3;
  queue->Send(data_ptr, data_sz);
}
// Verifies that the "user data" pointer is properly passed into proc callbacks.
TEST_F(ProcBuilderVisitorTest, GetsUserData) {
  const std::string kIrText = R"(
package p

chan c_i(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=none, metadata="")
chan c_o(bits[32], id=1, kind=streaming, ops=send_only, flow_control=none, metadata="")

proc the_proc(my_token: token, state: (), init=()) {
  literal.1: bits[32] = literal(value=3)
  receive.2: (token, bits[32]) = receive(my_token, channel_id=0)
  tuple_index.3: token = tuple_index(receive.2, index=0)
  tuple_index.4: bits[32] = tuple_index(receive.2, index=1)
  umul.5: bits[32] = umul(literal.1, tuple_index.4)
  send.6: token = send(tuple_index.3, umul.5, channel_id=1)
  next (send.6, state)
}
)";

  XLS_ASSERT_OK(InitPackage(kIrText));
  auto module = std::make_unique<llvm::Module>(kModuleName, ctx());
  XLS_ASSERT_OK(InitLlvm(module.get(), package()->GetTupleType({})));
  XLS_ASSERT_OK_AND_ASSIGN(auto xls_fn, package()->GetProc("the_proc"));

  XLS_ASSERT_OK_AND_ASSIGN(auto queue_mgr,
                           JitChannelQueueManager::Create(package()));
  EnqueueData(queue_mgr->GetQueueById(0).value(), 7);

  XLS_ASSERT_OK(ProcBuilderVisitor::Visit(
      module.get(), llvm_fn(), xls_fn, type_converter(),
      /*is_top=*/true, /*generate_packed=*/false, queue_mgr.get(),
      &GetsUserData_recv, &GetsUserData_send));

  // The provided JIT doesn't support ExecutionEngine::runFunction, so we have
  // to get the fn pointer and call that directly.
  auto fn = BuildEntryFn(std::move(module), kFunctionName);

  // We don't have any persistent state, so we don't reference params, hence
  // nullptr.
  uint64_t user_data = 7;
  fn(nullptr, nullptr, nullptr, absl::bit_cast<void*>(&user_data));
  EXPECT_EQ(DequeueData(queue_mgr->GetQueueById(1).value()), 21);
  EXPECT_EQ(user_data, 7 * 2 * 3);

  // Let's make sure we can call it 2x!
  EnqueueData(queue_mgr->GetQueueById(0).value(), 8);
  fn(nullptr, nullptr, nullptr, absl::bit_cast<void*>(&user_data));
  EXPECT_EQ(DequeueData(queue_mgr->GetQueueById(1).value()), 24);
  EXPECT_EQ(user_data, 7 * 2 * 3 * 2 * 3);
}

}  // namespace
}  // namespace xls
