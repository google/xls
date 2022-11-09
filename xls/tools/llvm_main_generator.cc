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

// llvm_main_generator generates a "main" (as in int main(argc, char** argv))
// function for an LLVM IR sample, such as that produced by running the fuzzer.
// This allows the sample IR to be run by llc or lli _WITHOUT_MODIFICATION_ as
// produced by the JIT. Without this, head-to-head comparisons have only been
// possible by modifying the IR, which has been _very_ troublesome in the past.
//
// The main() function operates as follows (and is generated to do such):
//  - The size of the input alloca is determined by examining the args passed
//    to the program; argv[0] is the path to the sample, and the remaining
//    are arguments in XLS Value textual form (via XlsJitGetArgBufferSize().
//  - The alloca is created and packed with the command-line args (via
//    PackArgs()).
//  - An alloca is created with the size of the output type (taken from the
//    sample's signature).
//  - The sample is invoked with the allocas above.
//  - The result is extracted from the output buffer and printed to the terminal
//    (in XLS value textual form) via UnpackAndPrintBuffer().
#include <string>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "llvm/include/llvm/IR/DerivedTypes.h"
#include "llvm/include/llvm/IR/Function.h"
#include "llvm/include/llvm/IR/IRBuilder.h"
#include "llvm/include/llvm/IR/LLVMContext.h"
#include "llvm/include/llvm/IR/Module.h"
#include "llvm/include/llvm/IRReader/IRReader.h"
#include "llvm/include/llvm/Support/SourceMgr.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/jit/llvm_type_converter.h"
#include "xls/jit/orc_jit.h"

ABSL_FLAG(
    std::string, entry_function, "sample::__sample__main",
    "Entry function to generate a \"main\" for (as mangled in the LLVM IR.");
ABSL_FLAG(std::string, input, "", "Path to input LLVM IR file.");
ABSL_FLAG(std::string, output, "", "Path at which to write the output.");
ABSL_FLAG(std::string, output_type, "",
          "Type of the function output (expressed as an XLS IR type).");

namespace xls {

// Converts the given LLVM type into a corresponding XLS type.
using TypeStore = absl::flat_hash_map<llvm::Type*, std::unique_ptr<Type>>;

absl::Status RealMain(const std::string& input_path,
                      const std::string& output_path,
                      const std::string& entry_function_name,
                      const std::string& xls_output_type_string) {
  // 1. Load LLVM IR.
  llvm::LLVMContext context;
  llvm::SMDiagnostic parse_error;
  std::unique_ptr<llvm::Module> module =
      llvm::parseIRFile(input_path, parse_error, context);
  if (module == nullptr) {
    XLS_LOG(INFO) << parse_error.getMessage().str();
    return absl::InvalidArgumentError("Could not parse input LLVM IR!");
  }
  llvm::Function* entry_function = module->getFunction(entry_function_name);
  XLS_CHECK(entry_function != nullptr);

  // 2. Create all the objects needed to create our "main".
  // Common LLVM type decls.
  llvm::Type* int8_type = llvm::IntegerType::get(context, 8);
  llvm::Type* argv_ptr_type = llvm::PointerType::get(
      llvm::PointerType::get(int8_type, /*AddressSpace=*/0),
      /*AddressSpace=*/0);
  llvm::Type* int32_type = llvm::IntegerType::get(context, 32);
  llvm::Type* int64_type = llvm::IntegerType::get(context, 64);

  std::vector<llvm::Type*> main_param_types({int32_type, argv_ptr_type});
  llvm::FunctionType* main_type =
      llvm::FunctionType::get(int32_type, main_param_types, /*isVarArg=*/false);
  llvm::Function* main_function = llvm::cast<llvm::Function>(
      module->getOrInsertFunction("main", main_type).getCallee());
  llvm::BasicBlock* block =
      llvm::BasicBlock::Create(context, "block_main", main_function);
  llvm::IRBuilder<> builder(block);

  // 3. Now, in LLVM-space:
  // 3a. Determine the size of the alloca.
  llvm::FunctionType* get_size_type =
      llvm::FunctionType::get(int64_type, main_param_types, /*isVarArg=*/false);
  llvm::Function* get_size_function = llvm::Function::Create(
      get_size_type, llvm::GlobalValue::LinkageTypes::ExternalLinkage,
      "XlsJitGetArgBufferSize", module.get());
  std::vector<llvm::Value*> get_size_args({
      main_function->getArg(0),
      main_function->getArg(1),
  });
  llvm::CallInst* get_size_result =
      builder.CreateCall(get_size_type, get_size_function, get_size_args);

  // 3b. Create the big alloca.
  llvm::AllocaInst* input_alloca =
      builder.CreateAlloca(int8_type, get_size_result);

  // 3c. Now pack it.
  std::vector<llvm::Type*> pack_args_param_types{
      int32_type,
      argv_ptr_type,
      llvm::PointerType::get(int8_type, /*AddressSpace=*/0),
      int64_type,
  };
  llvm::FunctionType* pack_args_type = llvm::FunctionType::get(
      int64_type, pack_args_param_types, /*isVarArg=*/false);
  llvm::Function* pack_args_function = llvm::Function::Create(
      pack_args_type, llvm::GlobalValue::LinkageTypes::ExternalLinkage,
      "XlsJitPackArgs", module.get());
  std::vector<llvm::Value*> pack_args_args({main_function->getArg(0),
                                            main_function->getArg(1),
                                            input_alloca, get_size_result});
  builder.CreateCall(pack_args_type, pack_args_function, pack_args_args);

  // 4. Now capture the program output.
  // 4a. Get the output type of the entry function.
  Package package("the_package");
  XLS_ASSIGN_OR_RETURN(Type * xls_output_type,
                       Parser::ParseType(xls_output_type_string, &package));
  XLS_ASSIGN_OR_RETURN(llvm::DataLayout data_layout,
                       OrcJit::CreateDataLayout());

  LlvmTypeConverter type_converter(&context, data_layout);
  llvm::Type* base_output_type =
      type_converter.ConvertToLlvmType(xls_output_type);
  llvm::Type* output_type =
      llvm::PointerType::get(base_output_type, /*AddressSpace=*/0u);

  // 4b. Now create an alloca for it. The entry function will populate this as
  // its second-to-last instruction.
  llvm::AllocaInst* output_alloca =
      builder.CreateAlloca(output_type, /*AddrSpace=*/0u);

  // 5. Call the entry function!
  builder.CreateCall(entry_function->getFunctionType(), entry_function,
                     {input_alloca, output_alloca});

  // 6. Finally, unpack and print the output.
  // To avoid having to list the output type on the command line, we:
  //  - Store the XLS type string as a constant in the program.
  //  - Pass that type string into UnpackAndPrint to parse & use.
  TypeStore type_store;
  llvm::Constant* type_as_string =
      llvm::ConstantDataArray::getString(context, xls_output_type_string,
                                         /*AddNull=*/true);
  llvm::AllocaInst* output_type_string_alloca =
      builder.CreateAlloca(type_as_string->getType());
  builder.CreateStore(type_as_string, output_type_string_alloca);

  std::vector<llvm::Type*> unpack_param_types{
      output_type_string_alloca->getType(),
      int32_type,
      argv_ptr_type,
      llvm::PointerType::get(int8_type, /*AddressSpace=*/0),
  };
  llvm::FunctionType* unpack_function_type = llvm::FunctionType::get(
      int32_type, unpack_param_types, /*isVarArg=*/false);
  llvm::Function* unpack_function = llvm::Function::Create(
      unpack_function_type, llvm::GlobalValue::LinkageTypes::ExternalLinkage,
      "XlsJitUnpackAndPrintBuffer", module.get());
  llvm::Value* cast_alloca =
      builder.CreateBitCast(output_alloca, input_alloca->getType());
  std::vector<llvm::Value*> unpack_args({
      output_type_string_alloca,
      main_function->getArg(0),
      main_function->getArg(1),
      cast_alloca,
  });

  builder.CreateCall(unpack_function_type, unpack_function, unpack_args);
  builder.CreateRet(llvm::ConstantInt::get(int32_type, 0));

  // FINALLY, output the new IR to the output.
  return xls::SetFileContents(output_path, DumpLlvmModuleToString(*module));
}

}  // namespace xls

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);

  if (absl::GetFlag(FLAGS_input).empty()) {
    XLS_LOG(INFO) << "--input cannot be empty!";
    return 1;
  }

  if (absl::GetFlag(FLAGS_output).empty()) {
    XLS_LOG(INFO) << "--output cannot be empty!";
    return 1;
  }

  if (absl::GetFlag(FLAGS_entry_function).empty() ||
      absl::GetFlag(FLAGS_entry_function) == "main") {
    XLS_LOG(INFO) << "--entry function cannot be empty or \"main\"!";
    return 1;
  }

  XLS_QCHECK(!absl::GetFlag(FLAGS_output_type).empty())
      << "--output_type cannot be empty.";

  XLS_QCHECK_OK(xls::RealMain(
      absl::GetFlag(FLAGS_input), absl::GetFlag(FLAGS_output),
      absl::GetFlag(FLAGS_entry_function), absl::GetFlag(FLAGS_output_type)));
  return 0;
}
