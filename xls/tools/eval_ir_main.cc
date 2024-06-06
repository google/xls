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

#include <algorithm>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/ADT/APInt.h"
#include "llvm/include/llvm/ADT/StringRef.h"
#include "llvm/include/llvm/CodeGen/CommandFlags.h"
#include "llvm/include/llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/include/llvm/ExecutionEngine/GenericValue.h"
#include "llvm/include/llvm/ExecutionEngine/Interpreter.h"  // IWYU pragma: keep
#include "llvm/include/llvm/IR/Attributes.h"
#include "llvm/include/llvm/IR/BasicBlock.h"
#include "llvm/include/llvm/IR/Constant.h"
#include "llvm/include/llvm/IR/DataLayout.h"
#include "llvm/include/llvm/IR/DerivedTypes.h"
#include "llvm/include/llvm/IR/Function.h"
#include "llvm/include/llvm/IR/GlobalValue.h"
#include "llvm/include/llvm/IR/IRBuilder.h"
#include "llvm/include/llvm/IR/InlineAsm.h"
#include "llvm/include/llvm/IR/InstrTypes.h"
#include "llvm/include/llvm/IR/Instructions.h"
#include "llvm/include/llvm/IR/LLVMContext.h"
#include "llvm/include/llvm/IR/Module.h"
#include "llvm/include/llvm/IR/Type.h"
#include "llvm/include/llvm/IR/Value.h"
#include "llvm/include/llvm/IRReader/IRReader.h"
#include "llvm/include/llvm/Support/Alignment.h"
#include "llvm/include/llvm/Support/Casting.h"
#include "llvm/include/llvm/Support/CodeGen.h"
#include "llvm/include/llvm/Support/ModRef.h"
#include "llvm/include/llvm/Support/SourceMgr.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "llvm/include/llvm/Target/TargetMachine.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_convert/conversion_info.h"
#include "xls/dslx/ir_convert/ir_converter.h"
#include "xls/dslx/mangle.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/warning_kind.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/interpreter/random_value.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/jit/function_jit.h"
#include "xls/jit/jit_buffer.h"
#include "xls/jit/llvm_type_converter.h"
#include "xls/jit/observer.h"
#include "xls/jit/orc_jit.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/passes/pass_base.h"

static constexpr std::string_view kUsage = R"(
Evaluates an IR file with user-specified or random inputs using the IR
interpreter. Example invocations:

Evaluate IR with a single input:

   eval_ir_main --input='bits[32]:42; (bits[7]:0, bits[20]:4)' IR_FILE

Evaluate IR with a single input and fail if result does not match expected
value:

   eval_ir_main --input='bits[32]:42; (bits[7]:0, bits[20]:4)' \
      --expected='bits[3]:5' IR_FILE

Evaluate an IR function with a batch of arguments, one per line in INPUT_FILE
and compare each result against a corresponding values in EXPECTED_FILE:

   eval_ir_main --input_file=INPUT_FILE --expected_file=EXPECTED_FILE IR_FILE

Evaluate IR with randomly generated inputs:

   eval_ir_main --random_inputs=100 IR_FILE

Evaluate IR with randomly-generated inputs, guaranteed to be odd:

   eval_ir_main --random_inputs=100 \
       --input_validator="fn validator(x: u32) -> bool { (x & u32:1) as bool }"

Evaluate IR with randomly-generated inputs, whose constraints are specified
in foo.x (with a function named "validator"):

   eval_ir_main --random_inputs=100 --input_validator_path=foo.x

Evaluate IR before and after optimizations:

   eval_ir_main --random_inputs=100 --optimize_ir IR_FILE

Evaluate IR after every optimization pass:

   eval_ir_main --random_inputs=100 --optimize_ir --eval_after_each_pass IR_FILE

Evaluate IR using the JIT and with the interpreter and compare the results:

   eval_ir_main --test_llvm_jit --random_inputs=100  IR_FILE
)";

// LINT.IfChange
ABSL_FLAG(std::string, top, "", "Top entity to evaluate.");
ABSL_FLAG(std::string, input, "",
          "The input to the function as a semicolon-separated list of typed "
          "values. For example: \"bits[32]:42; (bits[7]:0, bits[20]:4)\"");
ABSL_FLAG(std::string, input_file, "",
          "Inputs to interpreter, one set per line. Each line should contain a "
          "semicolon-separated set of typed values. Cannot be specified with "
          "--input.");
ABSL_FLAG(int64_t, random_inputs, 0,
          "If non-zero, this is the number of randomly generated inputs to use "
          "in evaluation. Cannot be specified with --input.");
ABSL_FLAG(std::string, expected, "",
          "The expected result of the evaluation. A non-zero error code is "
          "returned if the evaluated result does not match.");
ABSL_FLAG(
    std::string, expected_file, "",
    "The expected result(s) of the evaluation(s). A non-zero error code is "
    "returned if the evaluated result does not match. Must be specified with "
    "--input_file.");
ABSL_FLAG(bool, optimize_ir, false,
          "Run optimization passes on the input and evaluate before and after "
          "optimizations. A non-zero error status is returned if the results "
          "do not match.");
ABSL_FLAG(
    bool, eval_after_each_pass, false,
    "When specified with --optimize_ir, run evaluation after each pass. "
    "A non-zero error status is returned if any of the results do not match.");
ABSL_FLAG(bool, use_llvm_jit, true, "Use the LLVM IR JIT for execution.");
ABSL_FLAG(bool, test_llvm_jit, false,
          "If true, then run the JIT and compare the results against the "
          "interpereter.");
ABSL_FLAG(int64_t, llvm_opt_level, 3,
          "The optimization level of the LLVM JIT. Valid values are from 0 (no "
          "optimizations) to 3 (maximum optimizations).");
ABSL_FLAG(std::string, input_validator_expr, "",
          "DSLX expression to validate randomly-generated inputs. "
          "The expression can reference entry function input arguments "
          "and should return true if the arguments are valid for the "
          "function and false otherwise.");
ABSL_FLAG(std::string, input_validator_path, "",
          "Path to a file containing DSLX for an input validator as with "
          "the `--input_validator` flag.");
ABSL_FLAG(std::string, dslx_stdlib_path, xls::kDefaultDslxStdlibPath,
          "Path to DSLX standard library");
ABSL_FLAG(std::string, dslx_path, "",
          "Additional paths to search for modules (colon delimited).");
ABSL_FLAG(int64_t, input_validator_limit, 1024,
          "Maximum number of tries to generate a valid random input before "
          "giving up. Only used if \"input_validator\" is set.");

ABSL_FLAG(
    std::string, test_only_inject_jit_result, "",
    "Test-only flag for injecting the result produced by the JIT. Used to "
    "force mismatches between JIT and interpreter for testing purposed.");
// LINT.ThenChange(//xls/build_rules/xls_ir_rules.bzl)

ABSL_FLAG(std::optional<std::string>, llvm_jit_ir_output, std::nullopt,
          "Path to write the (unoptimized) LLVM IR which the JIT generates.");
ABSL_FLAG(std::optional<std::string>, llvm_jit_opt_ir_output, std::nullopt,
          "Path to write the (optimized) LLVM IR which the JIT generates.");
ABSL_FLAG(std::optional<std::string>, llvm_jit_asm_output, std::nullopt,
          "Path to write the assembly code which the JIT generates.");
ABSL_FLAG(
    std::optional<std::string>, llvm_jit_main_wrapper_output, std::nullopt,
    "Path to write a simple LLVM IR program to which invokes the jit "
    "code. (The program writes output to fd-1). --use_llvm_jit must be true.");
ABSL_FLAG(bool, llvm_jit_main_wrapper_write_is_linked, false,
          "Make the main wrapper call the write libc function instead of just "
          "doing a volatile memmove, incompatible with interpreter.");
// TODO(allight): It would be nice to enable doing this automatically if the
// llvm jit code crashes or something.
ABSL_FLAG(
    bool, use_llvm_jit_interpreter, false,
    "Instead of compiling jitted XLS ir code and executing it, compile it to "
    "LLVM ir and then interpret the LLVM IR. --use_llvm_jit must be true. Use "
    "--llvm_opt_level=0 if you want to execute the unoptimized llvm ir.");

namespace xls {
namespace {

// Name of the dummy package created to hold the validator function, if any.
constexpr std::string_view kPackageName = "validator";

// Encapsulates a set of arguments to pass to the function for evaluation and
// the expected result.
struct ArgSet {
  std::vector<Value> args;
  std::optional<Value> expected;
};

// Returns the given arguments as a semicolon-separated string.
std::string ArgsToString(absl::Span<const Value> args) {
  return absl::StrJoin(args, "; ", ValueFormatterHex);
}

class EvalIrJitObserver final : public JitObserver {
 public:
  EvalIrJitObserver(std::optional<std::string> ir,
                    std::optional<std::string> opt_ir,
                    bool interpreter,
                    std::optional<std::string> assembly)
      : ir_(std::move(ir)),
        opt_ir_(std::move(opt_ir)),
        interpreter_(interpreter),
        assembly_(std::move(assembly)) {}
  std::string_view saved_opt_ir() const { return saved_opt_ir_; }
  JitObserverRequests GetNotificationOptions() const final {
    return JitObserverRequests{
        .unoptimized_module = ir_.has_value(),
        .optimized_module = opt_ir_.has_value() || interpreter_,
        .assembly_code_str = assembly_.has_value()};
  }

  void UnoptimizedModule(const llvm::Module* module) final {
    if (ir_) {
      std::string buffer;
      llvm::raw_string_ostream ostream(buffer);
      module->print(ostream, nullptr);
      ostream.flush();
      CHECK_OK(SetFileContents(*ir_, buffer));
    }
  }
  void OptimizedModule(const llvm::Module* module) final {
    if (absl::GetFlag(FLAGS_use_llvm_jit_interpreter) || opt_ir_) {
      llvm::raw_string_ostream ostream(saved_opt_ir_);
      module->print(ostream, nullptr);
      ostream.flush();
      if (opt_ir_) {
        CHECK_OK(SetFileContents(*opt_ir_, saved_opt_ir_));
      }
    }
  }
  void AssemblyCodeString(const llvm::Module* module,
                          std::string_view asm_code) final {
    if (assembly_) {
      CHECK_OK(SetFileContents(*assembly_, asm_code));
    }
  }

 private:
  std::optional<std::filesystem::path> ir_;
  std::optional<std::filesystem::path> opt_ir_;
  bool interpreter_;
  std::optional<std::filesystem::path> assembly_;
  std::string saved_opt_ir_;
};

absl::StatusOr<InterpreterResult<Value>> RunLlvmInterpreter(
    std::string_view llvm_ir, FunctionJit* jit, absl::Span<const Value> args) {
  llvm::SMDiagnostic diag;
  llvm::LLVMContext ctx;
  std::string ir_id =
      absl::StrFormat("in-memory-data-for-%s", jit->GetJittedFunctionName());
  std::unique_ptr<llvm::Module> mod(
      llvm::parseIR(llvm::MemoryBufferRef(llvm_ir, ir_id), diag, ctx));

  if (mod == nullptr) {
    std::string buffer;
    llvm::raw_string_ostream ostream(buffer);
    diag.print("eval_ir_main___llvm_jit_interpreter", ostream,
               /*ShowColors=*/false);
    ostream.flush();
    return absl::InternalError(absl::StrFormat(
        "Failed to initialize llvm interpreter. Error was:\n%s", buffer));
  }

  std::string error;
  llvm::EngineBuilder builder(std::move(mod));
  // Whole point is to use an interpreter.
  builder.setEngineKind(llvm::EngineKind::Interpreter);
  builder.setVerifyModules(true);
  // Don't mess with the module code we give llvm, we've already compiled it.
  builder.setOptLevel(llvm::CodeGenOptLevel::None);
  builder.setErrorStr(&error);

  std::unique_ptr<llvm::ExecutionEngine> exec(builder.create());

  XLS_RET_CHECK(exec != nullptr)
      << "Failed to create llvm execution engine. Error was\n"
      << error;

  // Really don't compile anything!
  exec->DisableLazyCompilation();

  llvm::Function* function =
      exec->FindFunctionNamed(jit->GetJittedFunctionName());
  XLS_RET_CHECK(function != nullptr)
      << "Unable to find function named \"" << jit->GetJittedFunctionName()
      << "\" in module code\n"
      << llvm_ir;

  InterpreterEvents events;
  // Turn values into llvm values.
  // TODO(allight): Deduplicate with FunctionJit::Run
  std::vector<Type*> arg_types;
  arg_types.reserve(args.size());
  absl::c_transform(jit->function()->params(), std::back_inserter(arg_types),
                    [](const Param* p) { return p->GetType(); });
  JitTempBuffer temp_buffer(jit->jitted_function_base().CreateTempBuffer());
  JitArgumentSet input_set(jit->jitted_function_base().CreateInputBuffer());
  JitArgumentSet output_set(jit->jitted_function_base().CreateOutputBuffer());
  XLS_RETURN_IF_ERROR(
      jit->runtime()->PackArgs(args, arg_types, input_set.pointers()))
      << "Unable to pack arguments.";
  std::vector<llvm::GenericValue> llvm_arg_values(7);
  // input_ptrs
  llvm_arg_values[0].PointerVal = absl::bit_cast<void*>(input_set.get());
  // output_ptrs
  llvm_arg_values[1].PointerVal = absl::bit_cast<void*>(output_set.get());
  // tmp_buffer
  llvm_arg_values[2].PointerVal = temp_buffer.get();
  // events
  llvm_arg_values[3].PointerVal = &events;
  // user_data
  llvm_arg_values[4].PointerVal = nullptr;
  // runtime
  llvm_arg_values[5].PointerVal = jit->runtime();
  // continuation. This is a function so always 0.
  llvm_arg_values[6].IntVal =
      llvm::APInt(/*numBits=*/64, /*val=*/0, /*isSigned=*/true);

  // Run Static constructors (probably not really needed since our code doesn't
  // have any but might as well be future proof).
  exec->runStaticConstructorsDestructors(/*isDtors=*/false);
  // Run the interpreter
  // TODO(allight): It would be nice to have some better back tracing support
  // here since the interpreter has a pretty easy to read stack structure but
  // its private. Getting it while running this under gdb isn't that bad at
  // least.
  llvm::GenericValue result = exec->runFunction(function, llvm_arg_values);
  XLS_RET_CHECK_EQ(result.IntVal.getZExtValue(), 0)
      << "Function returned non-zero?";
  if (exec->hasError()) {
    return absl::InternalError(exec->getErrorMessage());
  }

  Value result_value = jit->runtime()->UnpackBuffer(
      output_set.pointers()[0], jit->function()->GetType()->return_type());
  return InterpreterResult<Value>{std::move(result_value), std::move(events)};
}

absl::Status WriteMainWrapper(Function* f, FunctionJit* jit,
                              absl::Span<const ArgSet> arg_sets_in,
                              const std::filesystem::path& output_path) {
  llvm::LLVMContext context;
  llvm::Module mod(absl::StrCat("wrapped_main_", jit->GetJittedFunctionName()),
                   context);
  // Technically we are not jitting but we do require that the ir be used on
  // exactly this machine.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<OrcJit> orc_jit, OrcJit::Create());
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<llvm::TargetMachine> machine,
      orc_jit->CreateTargetMachine());
  XLS_ASSIGN_OR_RETURN(llvm::DataLayout layout,
                       orc_jit->CreateDataLayout());
  LlvmTypeConverter type_convert(&context, layout);
  mod.setDataLayout(layout);
  mod.setTargetTriple(machine->getTargetTriple().str());

  llvm::Type* int8_type = llvm::IntegerType::get(context, 8);
  llvm::Type* argv_ptr_type =
      llvm::PointerType::get(
          llvm::PointerType::get(int8_type, /*AddressSpace=*/0),
          /*AddressSpace=*/0)
          ->getPointerTo();
  llvm::Type* raw_ptr_type =
      llvm::PointerType::get(context, /*AddressSpace=*/0);
  llvm::Type* int32_type = llvm::IntegerType::get(context, 32);
  llvm::Type* int64_type = llvm::IntegerType::get(context, 64);
  std::vector<llvm::Type*> main_param_types({int32_type, argv_ptr_type});
  llvm::FunctionType* main_type =
      llvm::FunctionType::get(int32_type, main_param_types, /*isVarArg=*/false);

  // Declare extern jit function.
  llvm::Function* entrypoint = llvm::Function::Create(
      llvm::FunctionType::get(int64_type,
                              {
                                  raw_ptr_type,  // input_ptrs
                                  raw_ptr_type,  // output_ptrs
                                  raw_ptr_type,  // tmp_buffer
                                  raw_ptr_type,  // events
                                  raw_ptr_type,  // user_data
                                  raw_ptr_type,  // runtime
                                  int64_type,    // continuation
                              },
                              /*isVarArg=*/false),
      llvm::Function::LinkageTypes::ExternalLinkage,
      jit->GetJittedFunctionName(), &mod);
  // A little function to call 'write' if it's linked or do a volatile
  // read/write if it's not to prevent optimization shennanigans.
  llvm::Function* write_output_func = llvm::Function::Create(
      llvm::FunctionType::get(int64_type, {raw_ptr_type}, /*isVarArg=*/false),
      llvm::Function::LinkageTypes::InternalLinkage, "write_output", &mod);
  llvm::BasicBlock* write_output_start = llvm::BasicBlock::Create(
      context, "write_output_func_start", write_output_func);
  llvm::IRBuilder<> write_start(write_output_start);
  if (absl::GetFlag(FLAGS_llvm_jit_main_wrapper_write_is_linked)) {
    // libc write. Under some llvm interpreter configs this won't be linked so
    // we need to be careful.
    llvm::Function* write_function = llvm::Function::Create(
        llvm::FunctionType::get(int64_type,
                                {
                                    int32_type,
                                    llvm::PointerType::get(int8_type, 0),
                                    int64_type,
                                },
                                /*isVarArg=*/false),
        llvm::Function::LinkageTypes::ExternalWeakLinkage, "write", &mod);
    write_start.CreateRet(write_start.CreateCall(
        write_function,
        {
            llvm::Constant::getIntegerValue(int32_type, llvm::APInt(32, 1)),
            write_output_func->getArg(0),
            llvm::Constant::getIntegerValue(
                int64_type, llvm::APInt(64, jit->GetReturnTypeSize())),
        }));
  } else {
    // We might as well do nothing here. The main's optnone will prevent llvm
    // from removing the calls to the test code.
    write_output_func->addFnAttrs(
        llvm::AttrBuilder(context)
            .addMemoryAttr(llvm::MemoryEffects::unknown())
            .addAttribute(llvm::Attribute::OptimizeNone)
            .addAttribute(llvm::Attribute::NoInline)
            .addAttribute(llvm::Attribute::MustProgress));
    write_start.CreateRet(
        llvm::Constant::getIntegerValue(int64_type, llvm::APInt(64, 0)));
  }

  auto alignment_of = [&](Type* type) -> llvm::Align {
    return layout.getPrefTypeAlign(type_convert.ConvertToLlvmType(type));
  };
  // Create the main function.
  llvm::Function* main_function = llvm::cast<llvm::Function>(
      mod.getOrInsertFunction("main", main_type).getCallee());
  // Make sure the optimizer won't just remove the calls to the test code.
  main_function->addFnAttrs(llvm::AttrBuilder(context)
                                .addAttribute(llvm::Attribute::OptimizeNone)
                                .addAttribute(llvm::Attribute::NoInline)
                                .addAttribute(llvm::Attribute::MustProgress));
  llvm::BasicBlock* block =
      llvm::BasicBlock::Create(context, "block_main", main_function);
  llvm::IRBuilder<> builder(block);
  // Setup input buffer.
  llvm::AllocaInst* input_ptrs = builder.CreateAlloca(
      llvm::ArrayType::get(llvm::PointerType::get(int8_type, 0),
                           f->params().size()),
      /*ArraySize=*/nullptr, /*Name=*/"input_ptrs");
  std::vector<std::vector<uint8_t>> input_buffers;
  std::vector<uint8_t*> input_buffers_ptrs;
  input_buffers.reserve(f->params().size());
  input_buffers_ptrs.reserve(f->params().size());
  for (int i = 0; i < f->params().size(); ++i) {
    input_buffers.emplace_back(jit->GetArgTypeSize(i), 0);
    input_buffers_ptrs.emplace_back(input_buffers.back().data());
  }

  // Setup output buffer.
  llvm::AllocaInst* output_buf = builder.CreateAlloca(
      llvm::ArrayType::get(int8_type, jit->GetReturnTypeSize()),
      /*ArraySize=*/nullptr, /*Name=*/"output_buffer");
  output_buf->setAlignment(alignment_of(f->GetType()->return_type()));
  llvm::AllocaInst* output_ptrs = builder.CreateAlloca(
      llvm::ArrayType::get(llvm::PointerType::get(int8_type, 0), 1),
      /*ArraySize=*/nullptr, /*Name=*/"output_ptrs");
  builder.CreateStore(output_buf, output_ptrs);
  // Setup tmp buffer
  llvm::AllocaInst* tmp_buffer = builder.CreateAlloca(
      llvm::ArrayType::get(int8_type,
                           std::max(int64_t{1}, jit->GetTempBufferSize())),
      /*ArraySize=*/nullptr, /*Name=*/"tmp_buffer");
  // events, user_data, runtime
  // All of these can be null (for functions) and we don't want to have to link
  // in the rest of the system.
  llvm::Constant* llvm_null =
      llvm::Constant::getNullValue(llvm::PointerType::get(context, 0));
  // continuation
  // Functions don't use this.
  llvm::Constant* continuation =
      llvm::Constant::getIntegerValue(int64_type, llvm::APInt(64, 0));
  std::vector<Type*> arg_types;
  arg_types.reserve(f->params().size());
  absl::c_transform(f->params(), std::back_inserter(arg_types),
                    [](Param* p) { return p->GetType(); });

  int32_t input_set = 0;
  for (const auto& set : arg_sets_in) {
    XLS_RETURN_IF_ERROR(
        jit->runtime()->PackArgs(set.args, arg_types, input_buffers_ptrs));
    for (int i = 0; i < f->params().size(); ++i) {
      std::string arg_name =
          absl::StrFormat("_input_set_%d_arg_%d_", input_set++, i);
      llvm::Constant* value = builder.CreateGlobalStringPtr(
          llvm::StringRef(reinterpret_cast<const char*>(input_buffers_ptrs[i]),
                          jit->GetArgTypeSize(i)),
          arg_name);
      mod.getNamedGlobal(arg_name)->setAlignment(
          alignment_of(f->GetType()->parameter_type(i)));

      builder.CreateStore(
          value, builder.CreateGEP(input_ptrs->getAllocatedType(), input_ptrs,
                                   {llvm::Constant::getIntegerValue(
                                        int32_type, llvm::APInt(32, 0)),
                                    llvm::Constant::getIntegerValue(
                                        int32_type, llvm::APInt(32, i))}));
    }

    // Call the function.
    builder.CreateCall(entrypoint->getFunctionType(), entrypoint,
                       {input_ptrs, output_ptrs, tmp_buffer, llvm_null,
                        llvm_null, llvm_null, continuation});
    // Write the output.
    builder.CreateCall(write_output_func->getFunctionType(), write_output_func,
                       {
                           output_buf,
                       });
  }
  builder.CreateRet(
      llvm::Constant::getIntegerValue(int32_type, llvm::APInt(32, 0)));

  // Write the built module.
  std::string buffer;
  llvm::raw_string_ostream ostream(buffer);
  mod.print(ostream, nullptr);
  ostream.flush();
  return SetFileContents(output_path, buffer);
}

// Evaluates the function with the given ArgSets. Returns an error if the result
// does not match expectations (if any). 'actual_src' and 'expected_src' are
// string descriptions of the sources of the actual results and expected
// results, respectively. These strings are included in error messages.
absl::StatusOr<std::vector<Value>> Eval(
    Function* f, absl::Span<const ArgSet> arg_sets, bool use_jit,
    std::string_view actual_src = "actual",
    std::string_view expected_src = "expected") {
  EvalIrJitObserver observer(absl::GetFlag(FLAGS_llvm_jit_ir_output),
                             absl::GetFlag(FLAGS_llvm_jit_opt_ir_output),
                             absl::GetFlag(FLAGS_use_llvm_jit_interpreter),
                             absl::GetFlag(FLAGS_llvm_jit_asm_output));
  std::unique_ptr<FunctionJit> jit;
  if (use_jit) {
    // No support for procs yet.
    XLS_ASSIGN_OR_RETURN(
        jit,
        FunctionJit::Create(f, absl::GetFlag(FLAGS_llvm_opt_level), &observer));
  }

  if (absl::GetFlag(FLAGS_llvm_jit_main_wrapper_output)) {
    // TODO(allight): Move this into aot_compiler...
    XLS_RETURN_IF_ERROR(WriteMainWrapper(
        f, jit.get(), arg_sets,
        std::filesystem::path(
            *absl::GetFlag(FLAGS_llvm_jit_main_wrapper_output))));
  }

  std::vector<Value> results;
  for (const ArgSet& arg_set : arg_sets) {
    Value result;
    if (use_jit) {
      if (absl::GetFlag(FLAGS_test_only_inject_jit_result).empty()) {
        if (absl::GetFlag(FLAGS_use_llvm_jit_interpreter)) {
          XLS_ASSIGN_OR_RETURN(
              result, DropInterpreterEvents(RunLlvmInterpreter(
                          observer.saved_opt_ir(), jit.get(), arg_set.args)));
        } else {
          XLS_ASSIGN_OR_RETURN(result,
                               DropInterpreterEvents(jit->Run(arg_set.args)));
        }
      } else {
        XLS_ASSIGN_OR_RETURN(result, Parser::ParseTypedValue(absl::GetFlag(
                                         FLAGS_test_only_inject_jit_result)));
      }
    } else {
      // TODO(https://github.com/google/xls/issues/506): 2021-10-12 Also compare
      // resulting events once the JIT fully supports events. Note: This will
      // require rethinking some of the control flow because event comparison
      // only makes sense for certain modes (optimize_ir and test_llvm_jit).
      XLS_ASSIGN_OR_RETURN(
          result, DropInterpreterEvents(InterpretFunction(f, arg_set.args)));
    }
    std::cout << result.ToString(FormatPreference::kHex) << '\n';

    if (arg_set.expected.has_value()) {
      if (result != *arg_set.expected) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Miscompare for input[%i] \"%s\"\n  %s: %s\n  %s: %s",
            results.size(), ArgsToString(arg_set.args), actual_src,
            result.ToString(FormatPreference::kHex), expected_src,
            arg_set.expected->ToString(FormatPreference::kHex)));
      }
    }
    results.push_back(result);
  }
  return results;
}

// An invariant checker which evaluates the entry function with the given
// ArgSets. Raises an error if expectations are not matched.
class EvalInvariantChecker : public OptimizationInvariantChecker {
 public:
  explicit EvalInvariantChecker(absl::Span<const ArgSet> arg_sets, bool use_jit)
      : arg_sets_(arg_sets.begin(), arg_sets.end()), use_jit_(use_jit) {}
  absl::Status Run(Package* package, const OptimizationPassOptions& options,
                   PassResults* results) const override {
    if (results->invocations.empty()) {
      std::cerr << "// Evaluating entry function at start of pipeline.\n";
    } else {
      std::cerr << "// Evaluating entry function after pass: "
                << results->invocations.back().pass_name << "\n";
    }
    XLS_ASSIGN_OR_RETURN(Function * f, package->GetTopAsFunction());
    XLS_RETURN_IF_ERROR(Eval(f, arg_sets_, use_jit_,
                             /*actual_src=*/results->invocations.empty()
                                 ? std::string("start of pipeline")
                                 : results->invocations.back().pass_name,
                             /*expected_src=*/"before optimizations")
                            .status());
    return absl::OkStatus();
  }

 private:
  std::vector<ArgSet> arg_sets_;
  bool use_jit_;
};

// Runs the given ArgSets through the given package. This includes optionally
// (based on flags) optimizing the IR and evaluating the ArgSets during and
// after optimizations.
absl::Status Run(Package* package, absl::Span<const ArgSet> arg_sets_in) {
  XLS_ASSIGN_OR_RETURN(Function * f, package->GetTopAsFunction());
  // Copy the input ArgSets because we want to write in expected values if they
  // do not exist.
  std::vector<ArgSet> arg_sets(arg_sets_in.begin(), arg_sets_in.end());

  if (absl::GetFlag(FLAGS_test_llvm_jit)) {
    QCHECK(!absl::GetFlag(FLAGS_optimize_ir))
        << "Cannot specify both --test_llvm_jit and --optimize_ir";
    XLS_ASSIGN_OR_RETURN(std::vector<Value> interpreter_results,
                         Eval(f, arg_sets, /*use_jit=*/false));
    for (int64_t i = 0; i < arg_sets.size(); ++i) {
      QCHECK(!arg_sets[i].expected.has_value())
          << "Cannot specify expected values when using --test_llvm_jit";
      arg_sets[i].expected = interpreter_results[i];
    }
    return Eval(f, arg_sets, /*use_jit=*/true, "JIT", "interpreter").status();
  }

  // Run the argsets through the IR before any optimizations. Write in the
  // results as the expected values if the expected value is not already
  // set. These expected values are used in any later evaluation after
  // optimizations.
  XLS_ASSIGN_OR_RETURN(std::vector<Value> results,
                       Eval(f, arg_sets, absl::GetFlag(FLAGS_use_llvm_jit)));
  for (int64_t i = 0; i < arg_sets.size(); ++i) {
    if (!arg_sets[i].expected.has_value()) {
      arg_sets[i].expected = results[i];
    }
  }

  // Run optimizations (optionally) and check the results against expectations
  // (either expected result passed in on the command line or the result
  // produced without optimizations).
  if (absl::GetFlag(FLAGS_optimize_ir)) {
    std::unique_ptr<OptimizationCompoundPass> pipeline =
        CreateOptimizationPassPipeline();
    if (absl::GetFlag(FLAGS_eval_after_each_pass)) {
      pipeline->AddInvariantChecker<EvalInvariantChecker>(
          arg_sets, absl::GetFlag(FLAGS_use_llvm_jit));
    }
    PassResults results;
    XLS_RETURN_IF_ERROR(
        pipeline->Run(package, OptimizationPassOptions(), &results).status());

    XLS_RETURN_IF_ERROR(Eval(f, arg_sets, absl::GetFlag(FLAGS_use_llvm_jit),
                             "after optimizations", "before optimizations")
                            .status());
  } else {
    XLS_RET_CHECK(!absl::GetFlag(FLAGS_eval_after_each_pass))
        << "Must specify --optimize_ir with --eval_after_each_pass";
  }
  return absl::OkStatus();
}

// Parse the given string as a semi-colon separated list of Values.
absl::StatusOr<ArgSet> ArgSetFromString(std::string_view args_string) {
  ArgSet arg_set;
  for (const std::string_view& value_string :
       absl::StrSplit(args_string, ';')) {
    XLS_ASSIGN_OR_RETURN(Value arg, Parser::ParseTypedValue(value_string));
    arg_set.args.push_back(arg);
  }
  return arg_set;
}

// Converts the given DSLX validation function into IR.
absl::StatusOr<std::unique_ptr<Package>> ConvertValidator(
    Function* f, std::string_view dslx_stdlib_path,
    absl::Span<const std::filesystem::path> dslx_paths,
    std::string_view validator_dslx) {
  dslx::ImportData import_data(dslx::CreateImportData(
      std::string(dslx_stdlib_path), dslx_paths, dslx::kDefaultWarningsSet));
  XLS_ASSIGN_OR_RETURN(dslx::TypecheckedModule module,
                       dslx::ParseAndTypecheck(validator_dslx, "fake_path",
                                               kPackageName, &import_data));
  XLS_ASSIGN_OR_RETURN(
      dslx::PackageConversionData data,
      dslx::ConvertModuleToPackage(module.module, &import_data, {}));
  std::unique_ptr<Package> package = std::move(data).package;
  XLS_ASSIGN_OR_RETURN(std::string mangled_name,
                       dslx::MangleDslxName(kPackageName, kPackageName,
                                            dslx::CallingConvention::kTypical));

  // Now verify that the validation function has the appropriate signature.
  XLS_ASSIGN_OR_RETURN(Function * validator,
                       package->GetFunction(mangled_name));
  Type* return_type = validator->return_value()->GetType();
  if (return_type->kind() != TypeKind::kBits ||
      return_type->AsBitsOrDie()->GetFlatBitCount() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Validator must return bits[1]; got %s", return_type->ToString()));
  }
  if (f->params().size() != validator->params().size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Validator has wrong number of params: %d vs. expected %d",
        validator->params().size(), f->params().size()));
  }
  for (int i = 0; i < f->params().size(); i++) {
    if (!f->param(i)->GetType()->IsEqualTo(validator->param(i)->GetType())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Validator arg %d mismatch.\n\tExpected: %s\n\tActual: %s", i,
          f->param(i)->GetType()->ToString(),
          validator->param(i)->GetType()->ToString()));
    }
  }
  return package;
}

// Runs the validator to confirm that the args set is compatible.
absl::StatusOr<bool> ValidateInput(Function* validator, const ArgSet& arg_set) {
  XLS_ASSIGN_OR_RETURN(Value result, DropInterpreterEvents(InterpretFunction(
                                         validator, arg_set.args)));
  XLS_ASSIGN_OR_RETURN(Bits bits, result.GetBitsWithStatus());
  XLS_RET_CHECK_EQ(bits.bit_count(), 1);
  return bits.IsOne();
}

absl::StatusOr<ArgSet> GenerateArgSet(Function* f, Function* validator,
                                      absl::BitGenRef rng) {
  ArgSet arg_set;
  if (validator == nullptr) {
    for (Param* param : f->params()) {
      arg_set.args.push_back(RandomValue(param->GetType(), rng));
    }
    return arg_set;
  }

  int input_validator_limit = absl::GetFlag(FLAGS_input_validator_limit);
  for (int i = 0; i < input_validator_limit; i++) {
    arg_set.args.clear();
    for (Param* param : f->params()) {
      arg_set.args.push_back(RandomValue(param->GetType(), rng));
    }

    XLS_ASSIGN_OR_RETURN(bool valid, ValidateInput(validator, arg_set));
    if (valid) {
      return arg_set;
    }
  }

  return absl::ResourceExhaustedError(absl::StrCat(
      "Unable to generate valid input after ", input_validator_limit,
      "attempts. The validator may be difficult/impossible to satisfy "
      "or -input_validator_limit should be increased."));
}

absl::Status RealMain(std::string_view input_path,
                      std::string_view dslx_stdlib_path,
                      absl::Span<const std::filesystem::path> dslx_paths) {
  if (input_path == "-") {
    input_path = "/dev/stdin";
  }
  XLS_ASSIGN_OR_RETURN(std::string contents, GetFileContents(input_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(contents, input_path));
  if (!absl::GetFlag(FLAGS_top).empty()) {
    XLS_RETURN_IF_ERROR(package->SetTopByName(absl::GetFlag(FLAGS_top)));
  }
  XLS_ASSIGN_OR_RETURN(Function * f, package->GetTopAsFunction());

  std::vector<ArgSet> arg_sets;
  if (!absl::GetFlag(FLAGS_input).empty()) {
    QCHECK_EQ(absl::GetFlag(FLAGS_random_inputs), 0)
        << "Cannot specify both --input and --random_inputs";
    QCHECK(absl::GetFlag(FLAGS_input_file).empty())
        << "Cannot specify both --input and --input_file";
    absl::StatusOr<ArgSet> arg_set_status =
        ArgSetFromString(absl::GetFlag(FLAGS_input));
    QCHECK_OK(arg_set_status.status())
        << "Failed to parse input: " << absl::GetFlag(FLAGS_input);

    arg_sets.push_back(arg_set_status.value());
  } else if (!absl::GetFlag(FLAGS_input_file).empty()) {
    QCHECK_EQ(absl::GetFlag(FLAGS_random_inputs), 0)
        << "Cannot specify both --input_file and --random_inputs";
    absl::StatusOr<std::string> args_input_file =
        GetFileContents(absl::GetFlag(FLAGS_input_file));
    QCHECK_OK(args_input_file.status());
    for (const auto& arg_line : absl::StrSplit(args_input_file.value(), '\n',
                                               absl::SkipWhitespace())) {
      absl::StatusOr<ArgSet> arg_set_status = ArgSetFromString(arg_line);
      QCHECK_OK(arg_set_status.status())
          << absl::StreamFormat("Invalid line in input file %s: %s",
                                absl::GetFlag(FLAGS_input_file), arg_line);
      arg_sets.push_back(arg_set_status.value());
    }
  } else {
    QCHECK_NE(absl::GetFlag(FLAGS_random_inputs), 0)
        << "Must specify --input, --input_file, or --random_inputs.";
    arg_sets.resize(absl::GetFlag(FLAGS_random_inputs));
    std::minstd_rand rng_engine;
    std::string validator_text = absl::GetFlag(FLAGS_input_validator_expr);
    std::filesystem::path validator_path =
        absl::GetFlag(FLAGS_input_validator_path);
    if (!validator_path.empty()) {
      XLS_ASSIGN_OR_RETURN(validator_text, GetFileContents(validator_path));
    }
    std::unique_ptr<Package> validator_pkg;
    Function* validator = nullptr;
    if (!validator_text.empty()) {
      XLS_ASSIGN_OR_RETURN(
          validator_pkg,
          ConvertValidator(f, dslx_stdlib_path, dslx_paths, validator_text));
      XLS_ASSIGN_OR_RETURN(
          std::string mangled_name,
          dslx::MangleDslxName(kPackageName, kPackageName,
                               dslx::CallingConvention::kTypical));
      XLS_ASSIGN_OR_RETURN(validator, validator_pkg->GetFunction(mangled_name));
    }

    for (ArgSet& arg_set : arg_sets) {
      XLS_ASSIGN_OR_RETURN(arg_set, GenerateArgSet(f, validator, rng_engine));
    }
  }

  if (!absl::GetFlag(FLAGS_expected).empty()) {
    QCHECK(absl::GetFlag(FLAGS_expected_file).empty())
        << "Cannot specify both --expected_file and --expected";
    absl::StatusOr<Value> expected_status =
        Parser::ParseTypedValue(absl::GetFlag(FLAGS_expected));
    QCHECK_OK(expected_status.status())
        << "Failed to parse expected value: " << absl::GetFlag(FLAGS_expected);
    // Set the expected value of every sample to 'expected'.
    for (ArgSet& arg_set : arg_sets) {
      arg_set.expected = expected_status.value();
    }
  } else if (!absl::GetFlag(FLAGS_expected_file).empty()) {
    absl::StatusOr<std::string> expected_file =
        GetFileContents(absl::GetFlag(FLAGS_expected_file));
    QCHECK_OK(expected_file.status());
    std::vector<Value> expecteds;
    for (const auto& expected_line :
         absl::StrSplit(expected_file.value(), '\n', absl::SkipWhitespace())) {
      absl::StatusOr<Value> expected_status =
          Parser::ParseTypedValue(expected_line);
      QCHECK_OK(expected_status.status())
          << absl::StreamFormat("Failed to parse line in expected file %s: %s",
                                expected_line, expected_line);
      expecteds.push_back(expected_status.value());
    }
    QCHECK_EQ(expecteds.size(), arg_sets.size())
        << "Number of values in expected file does not match the number of "
           "inputs.";
    for (int64_t i = 0; i < arg_sets.size(); ++i) {
      arg_sets[i].expected = expecteds[i];
    }
  }

  return Run(package.get(), arg_sets);
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);
  if (positional_arguments.empty()) {
    LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s <ir-path>",
                                      argv[0]);
  }
  QCHECK(absl::GetFlag(FLAGS_input_validator_expr).empty() ||
         absl::GetFlag(FLAGS_input_validator_path).empty())
      << "At most one one of 'input_validator' or 'input_validator_path' may "
         "be specified.";
  std::string dslx_stdlib_path = absl::GetFlag(FLAGS_dslx_stdlib_path);

  std::string dslx_path = absl::GetFlag(FLAGS_dslx_path);
  std::vector<std::string> dslx_path_strs = absl::StrSplit(dslx_path, ':');
  std::vector<std::filesystem::path> dslx_paths;
  dslx_paths.reserve(dslx_path_strs.size());
  for (const auto& path : dslx_path_strs) {
    dslx_paths.push_back(std::filesystem::path(path));
  }

  return xls::ExitStatus(
      xls::RealMain(positional_arguments[0], dslx_stdlib_path, dslx_paths));
}
