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

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "google/protobuf/text_format.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/fuzzer/sample_summary.pb.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"

const char kUsage[] = R"(

Appends a summary of a given IR file(s) to the specified Protobuf summary
file. emitted. The summary information includes information such as op types,
widths, etc. This is used by the fuzzer to give an indication of what kind of IR
operations are being covered. Usage:

  summarize_ir_main --optimized_ir=IR_FILE --unoptimized_ir=IR_FILE \
    --summary_file=SUMMARY_FILE

The summary file will be created if it does not exist. Otherwise the summary
file is appended to.
)";

ABSL_FLAG(std::string, optimized_ir, "", "Optimized IR file to summarize.");
ABSL_FLAG(std::string, summary_file, "", "Summary file to append to.");
ABSL_FLAG(
    std::string, timing, "",
    "A serialized fuzzer::SampleTimingProto to write into the summary file.");
ABSL_FLAG(std::string, unoptimized_ir, "", "Unoptimized IR file to summarize.");

namespace xls {
namespace {

std::string TypeToString(Type* type) {
  if (type->IsBits()) {
    return "bits";
  }
  if (type->IsArray()) {
    return "array";
  }
  if (type->IsTuple()) {
    return "tuple";
  }
  return "other";
}

void SummarizePackage(Package* package,
                      google::protobuf::RepeatedPtrField<fuzzer::NodeProto>* nodes) {
  for (const FunctionBase* fb : package->GetFunctionBases()) {
    for (Node* node : fb->nodes()) {
      fuzzer::NodeProto* node_proto = nodes->Add();
      node_proto->set_op(OpToString(node->op()));
      node_proto->set_type(TypeToString(node->GetType()));
      node_proto->set_width(node->GetType()->GetFlatBitCount());
      for (Node* operand : node->operands()) {
        fuzzer::NodeProto* operand_proto = node_proto->add_operands();
        operand_proto->set_op(OpToString(operand->op()));
        operand_proto->set_type(TypeToString(operand->GetType()));
        operand_proto->set_width(operand->GetType()->GetFlatBitCount());
      }
    }
  }
}
absl::StatusOr<std::unique_ptr<Package>> ParseFile(std::string_view path) {
  XLS_ASSIGN_OR_RETURN(std::string contents, GetFileContents(path));
  return Parser::ParsePackage(contents, path);
}

absl::Status RealMain(std::string_view unoptimized_path,
                      std::string_view optimized_path,
                      std::string_view timing_str) {
  fuzzer::SampleSummariesProto summaries;
  fuzzer::SampleSummaryProto* summary_proto = summaries.add_samples();

  if (!timing_str.empty()) {
    fuzzer::SampleTimingProto timing;
    google::protobuf::TextFormat::Parser parser;
    if (!parser.ParseFromString(std::string{timing_str}, &timing)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Failed to parse --timing flag value as "
                          "SampleTimingProto proto: %s",
                          timing_str));
    }
    *summary_proto->mutable_timing() = timing;
  }

  if (!unoptimized_path.empty()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                         ParseFile(unoptimized_path));
    SummarizePackage(package.get(), summary_proto->mutable_unoptimized_nodes());
  }
  if (!optimized_path.empty()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                         ParseFile(optimized_path));
    SummarizePackage(package.get(), summary_proto->mutable_optimized_nodes());
  }

  QCHECK(!absl::GetFlag(FLAGS_summary_file).empty())
      << "Must specify --summary_file.";
  // Since SampleSummariesProto contains just a repeated field, appending a new
  // such proto to a file that could potentially contain a SampleSummariesProto
  // already will make the file contain a valid SampleSummariesProto where
  // the prior repeated field and the one of `summaries` is concatenated. This
  // is essentially a quick and dirty RecordIO/Rigeli without the compression,
  // seeking and corruption handling features.
  XLS_RETURN_IF_ERROR(AppendStringToFile(absl::GetFlag(FLAGS_summary_file),
                                         summaries.SerializeAsString()));

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (!positional_arguments.empty()) {
    LOG(QFATAL) << "Usage:\n" << kUsage;
  }

  return xls::ExitStatus(xls::RealMain(absl::GetFlag(FLAGS_unoptimized_ir),
                                       absl::GetFlag(FLAGS_optimized_ir),
                                       absl::GetFlag(FLAGS_timing)));
}
