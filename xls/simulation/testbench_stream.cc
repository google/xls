// Copyright 2023 The XLS Authors
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

#include "xls/simulation/testbench_stream.h"

#include <cstdint>
#include <filesystem>  // NOLINT
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/vast.h"
#include "xls/common/file/named_pipe.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/thread.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/number_parser.h"
#include "xls/ir/source_location.h"

namespace xls {
namespace verilog {

/* static */ VastStreamEmitter VastStreamEmitter::Create(
    const TestbenchStream& stream, Module* m) {
  VastStreamEmitter emitter(stream);
  emitter.file_descriptor_ =
      m->AddInteger(absl::StrFormat("__%s_fd", stream.name), SourceInfo()),
  emitter.count_ =
      m->AddInteger(absl::StrFormat("__%s_cnt", stream.name), SourceInfo());
  emitter.errno_ =
      m->AddInteger(absl::StrFormat("__%s_errno", stream.name), SourceInfo());
  constexpr int64_t kStringSize = 256;
  emitter.error_string_ = m->AddReg(
      absl::StrFormat("__%s_error_str", stream.name),
      m->file()->BitVectorType(kStringSize * 8, SourceInfo()), SourceInfo());
  return emitter;
}

void VastStreamEmitter::EmitOpen(StatementBlock* block) const {
  // Emit code:
  //
  //   fd = $fopen(`__PIPE_NAME, "<mode>");
  //   if (fd == 0) begin
  //      errno = $ferror(fd, error_string);
  //      $display("FAILED: ...);
  //      $finish;
  //   end
  SystemFunctionCall* fopen_call = block->file()->Make<SystemFunctionCall>(
      SourceInfo(), "fopen",
      std::vector<Expression*>{
          block->file()->Make<MacroRef>(SourceInfo(), stream_.path_macro_name),
          block->file()->Make<QuotedString>(
              SourceInfo(),
              stream_.direction == TestbenchStreamDirection::kInput ? "r"
                                                                    : "w")});
  block->Add<BlockingAssignment>(SourceInfo(), file_descriptor_, fopen_call);
  Conditional* conditional = block->Add<Conditional>(
      SourceInfo(),
      block->file()->Equals(file_descriptor_,
                            block->file()->PlainLiteral(0, SourceInfo()),
                            SourceInfo()));
  SystemFunctionCall* ferror_call = block->file()->Make<SystemFunctionCall>(
      SourceInfo(), "ferror",
      std::vector<Expression*>{file_descriptor_, error_string_});
  conditional->consequent()->Add<BlockingAssignment>(SourceInfo(), errno_,
                                                     ferror_call);
  conditional->consequent()->Add<Display>(
      SourceInfo(),
      std::vector<Expression*>{
          block->file()->Make<QuotedString>(
              SourceInfo(),
              absl::StrFormat(
                  "FAILED: cannot open file for stream `%s` [errno %%d]: %%s",
                  stream_.name)),
          errno_, error_string_});
  conditional->consequent()->Add<Finish>(SourceInfo());
}

void VastStreamEmitter::EmitRead(StatementBlock* block, LogicRef* lhs) const {
  // Emit code:
  //
  //   cnt = $fscanf(fd, "%x\n", lhs);
  //   if (cnt == 0) begin
  //     $display("FAILED: ...");
  //     $finish;
  //   end
  SystemFunctionCall* call = block->file()->Make<SystemFunctionCall>(
      SourceInfo(), "fscanf",
      std::vector<Expression*>{
          file_descriptor_,
          block->file()->Make<QuotedString>(SourceInfo(), R"(%x\n)"), lhs});
  block->Add<BlockingAssignment>(SourceInfo(), count_, call);
  Conditional* conditional = block->Add<Conditional>(
      SourceInfo(),
      block->file()->Equals(
          count_, block->file()->PlainLiteral(0, SourceInfo()), SourceInfo()));
  conditional->consequent()->Add<Display>(
      SourceInfo(),
      std::vector<Expression*>{block->file()->Make<QuotedString>(
          SourceInfo(),
          absl::StrFormat("FAILED: $fscanf of file for stream `%s` failed.",
                          stream_.name))});
  conditional->consequent()->Add<Finish>(SourceInfo());
}

void VastStreamEmitter::EmitWrite(StatementBlock* block,
                                  Expression* value) const {
  // Emit code:
  //
  //   $fwriteh(fd, <value>);
  //   $fwrite(fd, "\n");
  block->Add<SystemTaskCall>(SourceInfo(), "fwriteh",
                             std::vector<Expression*>{file_descriptor_, value});
  block->Add<SystemTaskCall>(
      SourceInfo(), "fwrite",
      std::vector<Expression*>{
          file_descriptor_,
          block->file()->Make<QuotedString>(SourceInfo(), R"(\n)")});
}

void VastStreamEmitter::EmitClose(StatementBlock* block) const {
  block->Add<SystemTaskCall>(SourceInfo(), "fclose",
                             std::vector<Expression*>{file_descriptor_});
}

/* static */ absl::StatusOr<TestbenchStreamThread>
TestbenchStreamThread::Create(const TestbenchStream& stream,
                              const std::filesystem::path& named_pipe_path) {
  XLS_ASSIGN_OR_RETURN(NamedPipe named_pipe,
                       NamedPipe::Create(named_pipe_path));
  return TestbenchStreamThread(stream, std::move(named_pipe));
}

void TestbenchStreamThread::RunInputStream(
    TestbenchStreamThread::Producer producer) {
  VLOG(1) << absl::StrFormat("RunInputStream [%s]", stream_.name);
  thread_ = absl::WrapUnique(new Thread([this, producer]() {
    VLOG(1) << absl::StrFormat("Thread for stream `%s` started", stream_.name);
    absl::StatusOr<FileLineWriter> writer =
        FileLineWriter::Create(named_pipe_.path());
    if (!writer.ok()) {
      LOG(ERROR) << absl::StrFormat(
          "FileLineWriter creation failed for stream `%s`: %s", stream_.name,
          writer.status().message());
      MaybeSetError(writer.status());
      return;
    }
    while (true) {
      std::optional<Bits> bits = producer();
      if (!bits.has_value()) {
        VLOG(1) << absl::StrFormat(
            "Producer returned std::nullopt for stream `%s`", stream_.name);
        break;
      }
      VLOG(1) << absl::StrFormat("Value produced for stream `%s` : %s",
                                 stream_.name,
                                 BitsToString(*bits, FormatPreference::kHex));
      CHECK_EQ(bits->bit_count(), stream_.width);
      absl::Status write_status =
          writer->WriteLine(BitsToString(*bits, FormatPreference::kPlainHex));
      if (!write_status.ok()) {
        VLOG(1) << absl::StrFormat("Writing value to stream `%s` failed: %s",
                                   stream_.name, write_status.message());
        MaybeSetError(write_status);
        break;
      }
    }
  }));
}

void TestbenchStreamThread::RunOutputStream(
    TestbenchStreamThread::Consumer consumer) {
  VLOG(1) << absl::StrFormat("RunOutputStream [%s]", stream_.name);
  thread_ = absl::WrapUnique(new Thread([this, consumer]() {
    VLOG(1) << absl::StrFormat("Thread for stream `%s` started", stream_.name);
    absl::StatusOr<FileLineReader> reader =
        FileLineReader::Create(named_pipe_.path());
    if (!reader.ok()) {
      LOG(ERROR) << absl::StrFormat(
          "FileLineReader creation failed for stream `%s`: %s", stream_.name,
          reader.status().message());
      MaybeSetError(reader.status());
      return;
    }
    while (true) {
      absl::StatusOr<std::optional<std::string>> line = reader->ReadLine();
      if (!line.ok()) {
        LOG(ERROR) << absl::StrFormat("Error reading from stream `%s`: %s",
                                      stream_.name, line.status().message());
        MaybeSetError(line.status());
        break;
      }
      if (!line->has_value()) {
        // The other end of the pipe has been closed.
        VLOG(1) << absl::StrFormat(
            "Line reader for stream `%s` returned std::nullopt. Pipe has been "
            "closed.",
            stream_.name);
        break;
      }
      VLOG(1) << absl::StrFormat("Read from stream `%s`: %s", stream_.name,
                                 line->value());
      // TODO(meheff): 2023/11/8 Support capturing X values.
      if (absl::StrContains(line->value(), "x") ||
          absl::StrContains(line->value(), "X")) {
        LOG(ERROR) << absl::StrFormat("Stream `%s` produced an X value",
                                      stream_.name);
        MaybeSetError(absl::InvalidArgumentError(
            absl::StrFormat("Stream `%s` produced an X value: %s", stream_.name,
                            line->value())));
        continue;
      }
      absl::StatusOr<Bits> value = ParseUnsignedNumberWithoutPrefix(
          line->value(), FormatPreference::kHex,
          /*bit_count=*/stream_.width);
      if (!value.ok()) {
        LOG(ERROR) << absl::StrFormat(
            "Unabled to convert value from stream `%s` into Bits: %s",
            stream_.name, status_.message());
        MaybeSetError(value.status());
        continue;
      }
      absl::Status result = consumer(*value);
      if (!result.ok()) {
        VLOG(1) << absl::StrFormat(
            "Consumer for stream `%s` returned an error: %s", stream_.name,
            result.message());
        MaybeSetError(result);
        continue;
      }
    }
  }));
}

absl::Status TestbenchStreamThread::Join() {
  thread_->Join();
  return status_;
}

void TestbenchStreamThread::MaybeSetError(const absl::Status& status) {
  CHECK(!status.ok());
  if (status_.ok()) {
    status_ = status;
  }
}

}  // namespace verilog
}  // namespace xls
