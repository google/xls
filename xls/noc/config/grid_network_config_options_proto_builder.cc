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

#include "xls/noc/config/custom_network_config_builder_options_proto_builder.h"

namespace xls::noc {

GridNetworkConfigOptionsProtoBuilder&
GridNetworkConfigOptionsProtoBuilder::WithNumRows(int64_t num_rows) {
  proto_->set_num_rows(num_rows);
  return *this;
}

GridNetworkConfigOptionsProtoBuilder&
GridNetworkConfigOptionsProtoBuilder::WithNumColumns(int64_t num_columns) {
  proto_->set_num_columns(num_columns);
  return *this;
}

GridNetworkConfigOptionsProtoBuilder&
GridNetworkConfigOptionsProtoBuilder::WithRowLoopback(bool row_loopback) {
  proto_->set_row_loopback(row_loopback);
  return *this;
}

GridNetworkConfigOptionsProtoBuilder&
GridNetworkConfigOptionsProtoBuilder::WithColumnLoopback(bool column_loopback) {
  proto_->set_column_loopback(column_loopback);
  return *this;
}

}  // namespace xls::noc
