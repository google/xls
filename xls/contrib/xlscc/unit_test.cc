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

#include "xls/contrib/xlscc/unit_test.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using xls::status_testing::IsOkAndHolds;

void XlsccTestBase::Run(const absl::flat_hash_map<std::string, uint64_t>& args,
                        uint64_t expected, absl::string_view cpp_source,
                        xabsl::SourceLocation loc,
                        std::vector<absl::string_view> clang_argv) {
  testing::ScopedTrace trace(loc.file_name(), loc.line(), "Run failed");
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir,
                           SourceToIr(cpp_source, nullptr, clang_argv));
  RunAndExpectEq(args, expected, ir, false, false, loc);
}

void XlsccTestBase::Run(
    const absl::flat_hash_map<std::string, xls::Value>& args,
    xls::Value expected, absl::string_view cpp_source,
    xabsl::SourceLocation loc, std::vector<absl::string_view> clang_argv) {
  testing::ScopedTrace trace(loc.file_name(), loc.line(), "Run failed");
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir,
                           SourceToIr(cpp_source, nullptr, clang_argv));
  RunAndExpectEq(args, expected, ir, false, false, loc);
}

absl::Status XlsccTestBase::ScanFile(absl::string_view cpp_src,
                                     std::vector<absl::string_view> argv) {
  translator_ = absl::make_unique<xlscc::Translator>();

  return ScanFile(cpp_src, argv, translator_.get());
}

/* static */ absl::Status XlsccTestBase::ScanFile(
    absl::string_view cpp_src, std::vector<absl::string_view> argv,
    xlscc::Translator* translator) {
  XLS_ASSIGN_OR_RETURN(xls::TempFile temp,
                       xls::TempFile::CreateWithContent(cpp_src, ".cc"));

  std::string ps = temp.path();

  absl::Status ret;
  argv.push_back("-Werror");
  argv.push_back("-Wall");
  argv.push_back("-Wno-unknown-pragmas");
  XLS_RETURN_IF_ERROR(translator->SelectTop("my_package"));
  XLS_RETURN_IF_ERROR(translator->ScanFile(
      temp.path().c_str(), argv.empty()
                               ? absl::Span<absl::string_view>()
                               : absl::MakeSpan(&argv[0], argv.size())));
  return absl::OkStatus();
}

absl::StatusOr<std::string> XlsccTestBase::SourceToIr(
    absl::string_view cpp_src, xlscc::GeneratedFunction** pfunc,
    std::vector<absl::string_view> clang_argv) {
  XLS_RETURN_IF_ERROR(ScanFile(cpp_src, clang_argv));

  xls::Package package("my_package");
  XLS_ASSIGN_OR_RETURN(xlscc::GeneratedFunction * func,
                       translator_->GenerateIR_Top_Function(&package));
  if (pfunc) {
    *pfunc = func;
  }
  return package.DumpIr();
}

void XlsccTestBase::IOTest(std::string content, std::list<IOOpTest> inputs,
                           std::list<IOOpTest> outputs,
                           absl::flat_hash_map<std::string, xls::Value> args) {
  xlscc::GeneratedFunction* func;
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir_src, SourceToIr(content, &func));

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::Package> package,
                           ParsePackage(ir_src));
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * entry, package->EntryFunction());

  const int total_test_ops = inputs.size() + outputs.size();
  ASSERT_EQ(func->io_ops.size(), total_test_ops);

  std::list<IOOpTest> input_ops_orig = inputs;
  for (const xlscc::IOOp& op : func->io_ops) {
    if (op.op == xlscc::OpType::kRecv) {
      const IOOpTest test_op = inputs.front();
      inputs.pop_front();

      const std::string ch_name = op.channel->getNameAsString();
      XLS_CHECK_EQ(ch_name, test_op.name);

      auto new_val = xls::Value(xls::SBits(test_op.value, 32));

      if (!args.contains(ch_name)) {
        args[ch_name] = new_val;
        continue;
      }

      if (args[ch_name].IsBits()) {
        args[ch_name] = xls::Value::Tuple({args[ch_name], new_val});
      } else {
        XLS_CHECK(args[ch_name].IsTuple());
        const xls::Value prev_val = args[ch_name];
        XLS_ASSERT_OK_AND_ASSIGN(std::vector<xls::Value> values,
                                 prev_val.GetElements());
        values.push_back(new_val);
        args[ch_name] = xls::Value::Tuple(values);
      }
    }
  }

  XLS_ASSERT_OK_AND_ASSIGN(xls::Value actual,
                           xls::InterpretFunctionKwargs(entry, args));
  ASSERT_TRUE(actual.IsTuple());
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<xls::Value> returns,
                           actual.GetElements());
  ASSERT_EQ(returns.size(), total_test_ops);

  inputs = input_ops_orig;

  int op_idx = 0;
  for (const xlscc::IOOp& op : func->io_ops) {
    if (op.op == xlscc::OpType::kRecv) {
      const IOOpTest test_op = inputs.front();
      inputs.pop_front();

      const std::string ch_name = op.channel->getNameAsString();
      XLS_CHECK(ch_name == test_op.name);

      ASSERT_TRUE(returns[op_idx].IsBits());
      XLS_ASSERT_OK_AND_ASSIGN(uint64_t val, returns[op_idx].bits().ToUint64());
      ASSERT_EQ(val, test_op.condition ? 1 : 0);

    } else if (op.op == xlscc::OpType::kSend) {
      const IOOpTest test_op = outputs.front();
      outputs.pop_front();

      const std::string ch_name = op.channel->getNameAsString();
      XLS_CHECK(ch_name == test_op.name);

      ASSERT_TRUE(returns[op_idx].IsTuple());
      XLS_ASSERT_OK_AND_ASSIGN(std::vector<xls::Value> elements,
                               returns[op_idx].GetElements());
      ASSERT_EQ(elements.size(), 2);
      ASSERT_TRUE(elements[0].IsBits());
      ASSERT_TRUE(elements[1].IsBits());
      XLS_ASSERT_OK_AND_ASSIGN(uint64_t val0, elements[0].bits().ToUint64());
      XLS_ASSERT_OK_AND_ASSIGN(uint64_t val1, elements[1].bits().ToUint64());
      ASSERT_EQ(val1, test_op.condition ? 1 : 0);
      // Don't check data if it wasn't sent
      if (val1) {
        ASSERT_EQ(val0, test_op.value);
      }
    } else {
      FAIL() << "IOOp was neither send nor recv: " << static_cast<int>(op.op);
    }
    ++op_idx;
  }

  ASSERT_EQ(inputs.size(), 0);
  ASSERT_EQ(outputs.size(), 0);
}

void XlsccTestBase::ProcTest(
    std::string content, const xlscc::HLSBlock& block_spec,
    const absl::flat_hash_map<std::string, std::vector<xls::Value>>&
        inputs_by_channel,
    const absl::flat_hash_map<std::string, std::vector<xls::Value>>&
        outputs_by_channel) {
  XLS_ASSERT_OK(ScanFile(content));

  xls::Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Proc * proc,
                           translator_->GenerateIR_Block(&package, block_spec));

  std::vector<std::unique_ptr<xls::ChannelQueue>> queues;

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xls::ChannelQueueManager> queue_manager,
      xls::ChannelQueueManager::Create(/*user_defined_queues=*/{}, &package));

  // Enqueue all inputs.
  for (auto [ch_name, values] : inputs_by_channel) {
    XLS_ASSERT_OK_AND_ASSIGN(xls::ChannelQueue * queue,
                             queue_manager->GetQueueByName(ch_name));
    for (const xls::Value& value : values) {
      XLS_ASSERT_OK(queue->Enqueue(value));
    }
  }

  xls::ProcInterpreter interpreter(proc, queue_manager.get());
  ASSERT_THAT(
      interpreter.RunIterationUntilCompleteOrBlocked(),
      IsOkAndHolds(xls::ProcInterpreter::RunResult{.iteration_complete = true,
                                                   .progress_made = true,
                                                   .blocked_channels = {}}));

  for (auto [ch_name, values] : outputs_by_channel) {
    XLS_ASSERT_OK_AND_ASSIGN(xls::Channel * ch_out,
                             package.GetChannel(ch_name));
    xls::ChannelQueue& ch_out_queue = queue_manager->GetQueue(ch_out);

    EXPECT_THAT(values.size(), ch_out_queue.size());

    for (const xls::Value& value : values) {
      EXPECT_THAT(ch_out_queue.Dequeue(), IsOkAndHolds(value));
    }
  }
}
