// Copyright 2025 The XLS Authors
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

#ifndef XLS_DSLX_IR_CONVERT_PROC_SCOPED_CHANNEL_SCOPE_H_
#define XLS_DSLX_IR_CONVERT_PROC_SCOPED_CHANNEL_SCOPE_H_

#include <optional>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_convert/channel_scope.h"
#include "xls/dslx/ir_convert/conversion_info.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/type.h"

namespace xls::dslx {

class ProcScopedChannelScope : public GlobalChannelScope {
 public:
  ProcScopedChannelScope(
      PackageConversionData* conversion_info, ImportData* import_data,
      const ConvertOptions& options, ProcBuilder* proc_builder,
      std::optional<FifoConfig> default_fifo_config = std::nullopt)
      : GlobalChannelScope(conversion_info, import_data, options,
                           default_fifo_config),
        proc_builder_(proc_builder) {}

  ~ProcScopedChannelScope() override = default;

  absl::StatusOr<ChannelOrArray> DefineBoundaryChannelOrArray(
      const Param* param, TypeInfo* type_info,
      bool interface_channel = false) override;

 protected:
  absl::StatusOr<Channel*> CreateChannel(
      std::string_view name, ChannelOps ops, xls::Type* type,
      std::optional<ChannelConfig> channel_config) override;

 private:
  ProcBuilder* const proc_builder_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERT_PROC_SCOPED_CHANNEL_SCOPE_H_
