// Copyright 2024 The XLS Authors
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

#include <iostream>
#include <string_view>
#include <vector>

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "xls/common/init_xls.h"
#include "xls/ir/op_list.h"
#include "xls/ir/xls_op_name.pb.h"

namespace {
void add_op_entry(xls::OpNameRepresentationList& op_name_representation_list,
                  const char* cpp_enum_name, const char* ir_name) {
  xls::OpNameRepresentation* op_name_representation =
      op_name_representation_list.add_op_name_representations();
  op_name_representation->set_cpp_enum_name(cpp_enum_name);
  op_name_representation->set_ir_name(ir_name);
}

#define ENUM_TO_STRING(p) #p
#define ADD_OP_ENTRY(op_enum, b, op_name, d) \
  add_op_entry(op_name_representation_list, ENUM_TO_STRING(op_enum), op_name);
}  // namespace

static constexpr std::string_view kUsage = R"(
Generate a textproto file containing the OpNameRepresentationList proto, which
contains different name representations of every op.

Generate the textproto to stdout:
    generate_op_name_proto_main
)";

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  xls::OpNameRepresentationList op_name_representation_list;
  XLS_FOR_EACH_OP_TYPE(ADD_OP_ENTRY);

  google::protobuf::io::OstreamOutputStream cout(&std::cout);
  google::protobuf::TextFormat::Print(op_name_representation_list, &cout);

  return 0;
}
