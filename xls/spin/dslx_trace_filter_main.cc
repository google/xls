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

// Filters an EvaluatorResultsProto textproto to JSON channel events.
// Usage: dslx_trace_filter --input=trace.textproto [--output=out.json]
// [--format=hex]

#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_format.h"
#include "google/protobuf/text_format.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/evaluator_result.pb.h"

enum class TraceValueFormat { kHex, kDecimal, kBinary, kBase64 };

const std::string kHex = "hex";
const std::string kDecimal = "decimal";
const std::string kBinary = "binary";
const std::string kBase64 = "base64";
const std::string kUnknown = "unknown";

bool AbslParseFlag(absl::string_view text, TraceValueFormat* mode, std::string* error) {
  if (text == kHex)     { *mode = TraceValueFormat::kHex;     return true; }
  if (text == kDecimal) { *mode = TraceValueFormat::kDecimal; return true; }
  if (text == kBinary)  { *mode = TraceValueFormat::kBinary;  return true; }
  if (text == kBase64)  { *mode = TraceValueFormat::kBase64;  return true; }

  *error = absl::StrFormat("--format must be '%s', '%s', '%s', or '%s'.", kHex, kDecimal, kBinary, kBase64);
  return false;
}

std::string AbslUnparseFlag(TraceValueFormat mode) {
  switch (mode) {
    case TraceValueFormat::kHex:     return kHex;
    case TraceValueFormat::kDecimal: return kDecimal;
    case TraceValueFormat::kBinary:  return kBinary;
    case TraceValueFormat::kBase64:  return kBase64;
  }
  return kUnknown;
}

ABSL_FLAG(std::string, input, "",
          "Input EvaluatorResultsProto text proto file.");
ABSL_FLAG(std::string, output, "-",
          "Output file path, or '-' for stdout (default).");
ABSL_FLAG(TraceValueFormat, format, TraceValueFormat::kHex,
          "Value encoding: 'hex' (default), 'decimal', 'binary', or 'base64'.");

namespace xls::spin {
namespace {

// Encode the little-endian bytes of a BitsProto as a 0x-prefixed big-endian
// hex string padded to the full byte width.
std::string BitsToHex(const std::string& data) {
  std::string result = "0x";
  for (size_t i = data.size(); i-- > 0;) {
    // char is signed; cast to uint8_t to prevent sign extension.
    uint8_t byte_value = static_cast<uint8_t>(data[i]);
    absl::StrAppendFormat(&result, "%02x", byte_value);
  }
  return result;
}

// Encode little-endian bytes as a 0b-prefixed binary string, MSB first,
// trimmed to bit_count significant bits.
std::string BitsToBinary(const std::string& data, int64_t bit_count) {
  std::string result = "0b";
  for (int64_t i = bit_count - 1; i >= 0; --i) {
    int64_t byte_idx = i / 8;
    int64_t bit_idx = i % 8;
    char byte = (byte_idx < data.size()) ? data[byte_idx] : 0;
    result += ((byte >> bit_idx) & 1) ? '1' : '0';
  }
  return result;
}

// Decode little-endian bytes to an unsigned 64-bit integer.
uint64_t BitsToUint(const std::string& data) {
  uint64_t value = 0;
  for (size_t i = data.size(); i-- > 0;) {
    // char is signed; cast to uint8_t to prevent sign extension into value.
    uint8_t byte_value = static_cast<uint8_t>(data[i]);
    value = (value << 8) | byte_value;
  }
  return value;
}

// Format a ValueProto as a JSON object: {"bit_count": N, "data": <encoded>}.
std::string FormatValue(const ValueProto& value_proto,
                        const TraceValueFormat format) {
  if (!value_proto.has_bits()) {
    return "{\"bit_count\": 0, \"data\": \"(non-bits)\"}";
  }
  const std::string& data = value_proto.bits().data();
  int64_t bit_count = value_proto.bits().bit_count();
  std::string encoded;
  if (format == TraceValueFormat::kDecimal) {
    encoded = absl::StrFormat("%u", BitsToUint(data));
  } else if (format == TraceValueFormat::kBinary) {
    encoded = absl::StrFormat("\"%s\"", BitsToBinary(data, bit_count));
  } else if (format == TraceValueFormat::kBase64) {
    encoded = absl::StrFormat("\"%s\"", absl::Base64Escape(data));
  } else {
    encoded = absl::StrFormat("\"%s\"", BitsToHex(data));
  }
  return absl::StrFormat("{\"bit_count\": %d, \"data\": %s}", bit_count,
                         encoded);
}

std::string DirectionString(TraceChannelProto::Direction dir) {
  switch (dir) {
    case TraceChannelProto::SEND:
      return "SEND";
    case TraceChannelProto::RECV:
      return "RECV";
    default:
      return "UNKNOWN";
  }
}

std::string FormatTrace(const EvaluatorResultsProto& proto,
                        const TraceValueFormat format) {
  std::vector<std::string> entries;

  for (const EvaluatorResultProto& result : proto.results()) {
    for (const TraceMessageProto& trace_message : result.events().trace_msgs()) {
      if (!trace_message.has_channel()) {
        continue;
      }
      const TraceChannelProto& trace_channel = trace_message.channel();
      entries.push_back(absl::StrCat(
          "  {\"channel_name\": \"", trace_channel.channel_name(),
          "\", \"direction\": \"", DirectionString(trace_channel.direction()),
          "\", \"value\": ", FormatValue(trace_channel.value(), format), "}"));
    }
  }

  return absl::StrCat("[\n", absl::StrJoin(entries, ",\n"), "\n]\n");
}

absl::Status Run() {
  std::string input_path = absl::GetFlag(FLAGS_input);
  if (input_path.empty()) {
    return absl::InvalidArgumentError("--input is required.");
  }
  TraceValueFormat format = absl::GetFlag(FLAGS_format);

  LOG(INFO) << "Filtering DSLX trace from '" << input_path << "'";
  std::string content;
  XLS_ASSIGN_OR_RETURN(content, GetFileContents(input_path));

  EvaluatorResultsProto proto;
  if (!google::protobuf::TextFormat::ParseFromString(content, &proto)) {
    return absl::InvalidArgumentError("Failed to parse input text proto.");
  }

  std::string json = FormatTrace(proto, format);

  std::string output_path = absl::GetFlag(FLAGS_output);
  if (output_path == "-") {
    LOG(INFO) << "Writing filtered trace to stdout";
    std::cout << json;
  } else {
    LOG(INFO) << "Writing filtered trace to '" << output_path << "'";
    XLS_RETURN_IF_ERROR(SetFileContents(output_path, json));
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls::spin

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);
  return xls::ExitStatus(xls::spin::Run());
}
