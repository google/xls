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

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/ADT/SmallVector.h"
#include "llvm/include/llvm/Bitcode/BitcodeReader.h"
#include "llvm/include/llvm/Bitcode/BitcodeWriter.h"
#include "llvm/include/llvm/IR/Module.h"
#include "llvm/include/llvm/Support/Error.h"
#include "llvm/include/llvm/Support/MemoryBuffer.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "llvm/include/llvm/Transforms/Utils/SplitModule.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"

constexpr std::string_view kUsage = R"(
  Internal only build tool to generate segments from the unoptimized LLVM IR of
  an AOT compiled function.

  Usage:
    aot_compiler_generate_segments_main --input <llvm ir> --outputs <llvm ir> --private_salt <salt>
)";

ABSL_FLAG(std::string, input, "", "Input LLVM IR file.");
ABSL_FLAG(std::vector<std::string>, outputs, {},
          "Comma separated list of output LLVM IR files.");
ABSL_FLAG(std::string, private_salt, "",
          "Private salt to use to gensym the internal symbols shared between "
          "the segments.");

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
                      absl::Span<std::string const> output_file_paths,
                      std::string_view private_salt) {
  std::string salt =
      absl::StrCat(absl::StrReplaceAll(private_salt, {{"/", "__SLASH__"},
                                                      {"-", "_"},
                                                      {":", "__COLON__"},
                                                      {"@", "__AT__"}}),
                   "__");
  XLS_ASSIGN_OR_RETURN(std::string input_ir, GetFileContents(input_file_path));
  StrMemBuf input_buffer(input_ir);
  llvm::LLVMContext context;
  llvm::Expected<std::unique_ptr<llvm::Module>> module_or_err =
      llvm::parseBitcodeFile(input_buffer, context);
  if (llvm::Error error = module_or_err.takeError()) {
    return absl::InternalError(
        absl::StrCat("Failed to parse input bitcode: ", input_file_path,
                     " error: ", llvm::toString(std::move(error))));
  }
  std::unique_ptr<llvm::Module> module = std::move(*module_or_err);
  auto it = output_file_paths.begin();
  absl::Status status = absl::OkStatus();
  llvm::SplitModule(
      *module, output_file_paths.size(),
      [&](std::unique_ptr<llvm::Module> module) {
        for (auto& global : module->globals()) {
          if (global.hasName() &&
              global.getName().contains("__llvmsplit_unnamed")) {
            global.setName(absl::StrCat(global.getName().str(), "__", salt));
          }
        }
        llvm::SmallVector<char, 0> stream_buffer;
        llvm::raw_svector_ostream ostream(stream_buffer);
        llvm::WriteBitcodeToFile(*module, ostream);
        if (it == output_file_paths.end()) {
          status.Update(absl::InternalError(
              "More split modules than output file paths."));
          return;
        }
        status.Update(SetFileContents(
            *it, std::string_view(stream_buffer.begin(), stream_buffer.end())));
        ++it;
      });
  XLS_RET_CHECK(it == output_file_paths.end())
      << "Fewer split modules than output file paths.";
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (!positional_arguments.empty()) {
    LOG(QFATAL) << "Expected invocation: " << argv[0];
  }

  return xls::ExitStatus(xls::RealMain(absl::GetFlag(FLAGS_input),
                                       absl::GetFlag(FLAGS_outputs),
                                       absl::GetFlag(FLAGS_private_salt)));
}
