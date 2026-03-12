// Copyright 2026 The XLS Authors
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

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/include/llvm/Bitcode/BitcodeReader.h"
#include "llvm/include/llvm/IR/LLVMContext.h"
#include "llvm/include/llvm/Support/Error.h"
#include "llvm/include/llvm/Support/MemoryBuffer.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/jit/aot_compiler.h"
#include "xls/jit/jit_evaluator_options.h"

constexpr std::string_view kUsage = R"(
  Internal only build tool to compile an LLVM IR segment into an object file.
)";

ABSL_FLAG(std::string, input, "", "Input LLVM IR file.");
ABSL_FLAG(std::string, output_object, "", "Output object file.");
ABSL_FLAG(int64_t, llvm_opt_level, 3, "LLVM optimization level.");
#ifdef ABSL_HAVE_MEMORY_SANITIZER
static constexpr bool kHasMsan = true;
#else
static constexpr bool kHasMsan = false;
#endif
ABSL_FLAG(bool, include_msan, kHasMsan,
          "Whether to include MSAN instrumentation.");
ABSL_FLAG(bool, enable_llvm_coverage, false,
          "Whether to include llvm's 'trace-cmp' and 'inline-8bit-counters'"
          "coverage instrumentation");

namespace xls {
namespace {

class StrMemBuf : public llvm::MemoryBuffer {
 public:
  StrMemBuf(std::string_view str) : str_(str) {
    init(&*str.begin(), &*str.end(), /*RequiresNullTerminator=*/false);
  }

  llvm::MemoryBuffer::BufferKind getBufferKind() const override {
    return llvm::MemoryBuffer::MemoryBuffer_Malloc;
  }

 private:
  std::string_view str_;
};

absl::Status RealMain(std::string_view input_file_path,
                      std::string_view output_object_file,
                      int64_t llvm_opt_level, bool include_msan,
                      bool enable_llvm_coverage) {
  XLS_ASSIGN_OR_RETURN(std::string input_ir, GetFileContents(input_file_path));
  XLS_ASSIGN_OR_RETURN(
      auto compiler,
      AotCompiler::Create(JitEvaluatorOptions()
                              .set_opt_level(llvm_opt_level)
                              .set_include_msan(include_msan)
                              .set_enable_llvm_coverage(enable_llvm_coverage)));
  StrMemBuf input_buffer(input_ir);
  llvm::Expected<std::unique_ptr<llvm::Module>> module_or_err =
      llvm::parseBitcodeFile(input_buffer, *compiler->GetContext());
  if (llvm::Error error = module_or_err.takeError()) {
    return absl::InternalError(
        absl::StrCat("Failed to parse input bitcode: ", input_file_path,
                     " error: ", llvm::toString(std::move(error))));
  }
  std::unique_ptr<llvm::Module> module = std::move(*module_or_err);
  XLS_RETURN_IF_ERROR(compiler->CompileModule(std::move(module)));
  XLS_ASSIGN_OR_RETURN(std::vector<uint8_t> object_code,
                       std::move(compiler)->GetObjectCode());
  return SetFileContents(output_object_file, object_code);
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (!positional_arguments.empty()) {
    LOG(QFATAL) << "Expected invocation: " << argv[0];
  }

  return xls::ExitStatus(xls::RealMain(
      absl::GetFlag(FLAGS_input), absl::GetFlag(FLAGS_output_object),
      absl::GetFlag(FLAGS_llvm_opt_level), absl::GetFlag(FLAGS_include_msan),
      absl::GetFlag(FLAGS_enable_llvm_coverage)));
}
