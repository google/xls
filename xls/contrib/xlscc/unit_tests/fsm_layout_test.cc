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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "clang/include/clang/AST/Decl.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/generate_fsm.h"
#include "xls/contrib/xlscc/translator_types.h"
#include "xls/contrib/xlscc/unit_tests/unit_test.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"

namespace xlscc {

class FSMLayoutTest : public XlsccTestBase {
 public:
  absl::StatusOr<NewFSMLayout> GenerateTopFunction(std::string_view content) {
    generate_new_fsm_ = true;

    XLS_ASSIGN_OR_RETURN(xls::TempFile temp,
                         xls::TempFile::CreateWithContent(content, ".cc"));

    XLS_RETURN_IF_ERROR(
        ScanFile(temp, /* clang_argv= */ {}, /*io_test_mode=*/true,
                 /*error_on_init_interval=*/false,
                 /*error_on_uninitialized=*/false,
                 /*loc=*/xls::SourceLocation(),
                 /*fail_xlscc_check=*/false, /*max_unroll_iters=*/0));

    package_ = std::make_unique<xls::Package>("my_package");
    absl::flat_hash_map<const clang::NamedDecl*, xlscc::ChannelBundle>
        top_channel_injections = {};

    XLS_ASSIGN_OR_RETURN(xlscc::GeneratedFunction * func,
                         translator_->GenerateIR_Top_Function(
                             package_.get(), top_channel_injections));

    NewFSMGenerator generator(*translator_, DebugIrTraceFlags_FSMStates);

    return generator.LayoutNewFSM(*func, xls::SourceInfo());
  }

  std::vector<int64_t> FilterStates(
      const NewFSMLayout& layout, const int64_t find_slice_index,
      const absl::flat_hash_set<int64_t>& find_jumped_from_slice_indices) {
    std::vector<int64_t> ret;
    for (int64_t state_index = 0; state_index < layout.states.size();
         ++state_index) {
      const NewFSMState& state = layout.states.at(state_index);
      absl::flat_hash_set<int64_t> jumped_from_slice_indices;
      for (const JumpInfo& jump_info : state.jumped_from_slice_indices) {
        jumped_from_slice_indices.insert(jump_info.from_slice);
      }
      if (state.slice_index == find_slice_index &&
          jumped_from_slice_indices == find_jumped_from_slice_indices) {
        ret.push_back(state_index);
      }
    }
    return ret;
  }

  bool StateInputsDecl(const NewFSMLayout& layout, int64_t slice_index,
                       absl::flat_hash_set<int64_t> jumped_from_slice_indices,
                       std::string_view name,
                       std::optional<int64_t> from_slice_index = std::nullopt) {
    std::vector<int64_t> state_indices =
        FilterStates(layout, slice_index, jumped_from_slice_indices);
    EXPECT_EQ(state_indices.size(), 1L);
    const NewFSMState& state = layout.states.at(state_indices.front());
    for (const auto& [param, continuation_value] :
         state.current_inputs_by_input_param) {
      if (from_slice_index.has_value() &&
          from_slice_index.value() !=
              layout.output_slice_index_by_value.at(continuation_value)) {
        continue;
      }
      for (const auto& decl : continuation_value->decls) {
        if (decl->getNameAsString() == name) {
          return true;
        }
      }
    }
    return false;
  }

  bool StateSavesDecl(const NewFSMLayout& layout, int64_t slice_index,
                      absl::flat_hash_set<int64_t> jumped_from_slice_indices,
                      std::string_view name,
                      std::optional<int64_t> from_slice_index = std::nullopt) {
    std::vector<int64_t> state_indices =
        FilterStates(layout, slice_index, jumped_from_slice_indices);
    EXPECT_EQ(state_indices.size(), 1L);
    const NewFSMState& state = layout.states.at(state_indices.front());
    for (const ContinuationValue* value : state.values_to_save) {
      if (from_slice_index.has_value() &&
          from_slice_index.value() !=
              layout.output_slice_index_by_value.at(value)) {
        continue;
      }
      for (const auto& decl : value->decls) {
        if (decl->getNameAsString() == name) {
          return true;
        }
      }
    }

    return false;
  }
};

namespace {

TEST_F(FSMLayoutTest, Basic) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int x = in.read();
      const int y = in.read();
      out.write(y);
      out.write(x);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(NewFSMLayout layout, GenerateTopFunction(content));

  EXPECT_FALSE(StateInputsDecl(layout, /*slice_index=*/0,
                               /*jumped_from_slice_indices=*/{}, "x"));
  EXPECT_FALSE(StateInputsDecl(layout, /*slice_index=*/1,
                               /*jumped_from_slice_indices=*/{}, "x"));
  EXPECT_FALSE(StateInputsDecl(layout, /*slice_index=*/2,
                               /*jumped_from_slice_indices=*/{}, "x"));
  EXPECT_TRUE(StateInputsDecl(layout, /*slice_index=*/3,
                              /*jumped_from_slice_indices=*/{}, "x",
                              /*from_slice_index=*/1));
  EXPECT_FALSE(StateInputsDecl(layout, /*slice_index=*/4,
                               /*jumped_from_slice_indices=*/{}, "x"));

  EXPECT_FALSE(StateSavesDecl(layout, /*slice_index=*/0,
                              /*jumped_from_slice_indices=*/{}, "x"));
  EXPECT_TRUE(StateSavesDecl(layout, /*slice_index=*/1,
                             /*jumped_from_slice_indices=*/{}, "x",
                             /*from_slice_index=*/1));
  EXPECT_TRUE(StateSavesDecl(layout, /*slice_index=*/2,
                             /*jumped_from_slice_indices=*/{}, "x",
                             /*from_slice_index=*/1));
  EXPECT_FALSE(StateSavesDecl(layout, /*slice_index=*/3,
                              /*jumped_from_slice_indices=*/{}, "x"));
  EXPECT_FALSE(StateSavesDecl(layout, /*slice_index=*/4,
                              /*jumped_from_slice_indices=*/{}, "x"));
}

TEST_F(FSMLayoutTest, PipelinedLoop) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int r = in.read();
      int a = 0;
      #pragma hls_pipeline_init_interval 1
      for (int i=0;i<10;++i) {
        a += r;
      }
      out.write(a);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(NewFSMLayout layout, GenerateTopFunction(content));

  EXPECT_TRUE(StateInputsDecl(layout, /*slice_index=*/2,
                              /*jumped_from_slice_indices=*/{}, "r",
                              /*from_slice_index=*/1));
  EXPECT_TRUE(StateInputsDecl(layout, /*slice_index=*/2,
                              /*jumped_from_slice_indices=*/{}, "a",
                              /*from_slice_index=*/1));
  EXPECT_TRUE(StateInputsDecl(layout, /*slice_index=*/2,
                              /*jumped_from_slice_indices=*/{}, "i",
                              /*from_slice_index=*/1));

  EXPECT_TRUE(StateInputsDecl(layout, /*slice_index=*/2,
                              /*jumped_from_slice_indices=*/{2}, "r",
                              /*from_slice_index=*/1));
  EXPECT_TRUE(StateInputsDecl(layout, /*slice_index=*/2,
                              /*jumped_from_slice_indices=*/{2}, "a",
                              /*from_slice_index=*/2));
  EXPECT_TRUE(StateInputsDecl(layout, /*slice_index=*/2,
                              /*jumped_from_slice_indices=*/{2}, "i",
                              /*from_slice_index=*/2));

  EXPECT_TRUE(StateInputsDecl(layout, /*slice_index=*/3,
                              /*jumped_from_slice_indices=*/{}, "a",
                              /*from_slice_index=*/2));
  EXPECT_FALSE(StateInputsDecl(layout, /*slice_index=*/3,
                               /*jumped_from_slice_indices=*/{}, "r"));

  EXPECT_TRUE(StateSavesDecl(layout, /*slice_index=*/1,
                             /*jumped_from_slice_indices=*/{}, "a",
                             /*from_slice_index=*/1));
  EXPECT_TRUE(StateSavesDecl(layout, /*slice_index=*/1,
                             /*jumped_from_slice_indices=*/{}, "i",
                             /*from_slice_index=*/1));
  EXPECT_TRUE(StateSavesDecl(layout, /*slice_index=*/1,
                             /*jumped_from_slice_indices=*/{}, "r",
                             /*from_slice_index=*/1));

  EXPECT_TRUE(StateSavesDecl(layout, /*slice_index=*/2,
                             /*jumped_from_slice_indices=*/{}, "a",
                             /*from_slice_index=*/2));
  EXPECT_TRUE(StateSavesDecl(layout, /*slice_index=*/2,
                             /*jumped_from_slice_indices=*/{}, "i",
                             /*from_slice_index=*/2));
  EXPECT_TRUE(StateSavesDecl(layout, /*slice_index=*/2,
                             /*jumped_from_slice_indices=*/{}, "r",
                             /*from_slice_index=*/1));

  EXPECT_TRUE(StateSavesDecl(layout, /*slice_index=*/2,
                             /*jumped_from_slice_indices=*/{2}, "a",
                             /*from_slice_index=*/2));
  EXPECT_TRUE(StateSavesDecl(layout, /*slice_index=*/2,
                             /*jumped_from_slice_indices=*/{2}, "i",
                             /*from_slice_index=*/2));
  EXPECT_TRUE(StateSavesDecl(layout, /*slice_index=*/2,
                             /*jumped_from_slice_indices=*/{2}, "r",
                             /*from_slice_index=*/1));

  EXPECT_FALSE(StateSavesDecl(layout, /*slice_index=*/3,
                              /*jumped_from_slice_indices=*/{}, "r"));
}

TEST_F(FSMLayoutTest, PipelinedLoopBypass) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int r = in.read();
      int a = 0;
      #pragma hls_pipeline_init_interval 1
      for (int i=0;i<10;++i) {
        a += 5;
      }
      out.write(a + r);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(NewFSMLayout layout, GenerateTopFunction(content));

  EXPECT_FALSE(StateInputsDecl(layout, /*slice_index=*/2,
                               /*jumped_from_slice_indices=*/{}, "r"));
  EXPECT_FALSE(StateInputsDecl(layout, /*slice_index=*/2,
                               /*jumped_from_slice_indices=*/{2}, "r"));
  EXPECT_TRUE(StateInputsDecl(layout, /*slice_index=*/3,
                              /*jumped_from_slice_indices=*/{}, "r",
                              /*from_slice_index=*/1));

  EXPECT_TRUE(StateSavesDecl(layout, /*slice_index=*/1,
                             /*jumped_from_slice_indices=*/{}, "r",
                             /*from_slice_index=*/1));
  EXPECT_TRUE(StateSavesDecl(layout, /*slice_index=*/2,
                             /*jumped_from_slice_indices=*/{}, "r",
                             /*from_slice_index=*/1));
  EXPECT_TRUE(StateSavesDecl(layout, /*slice_index=*/2,
                             /*jumped_from_slice_indices=*/{2}, "r",
                             /*from_slice_index=*/1));

  EXPECT_FALSE(StateSavesDecl(layout, /*slice_index=*/3,
                              /*jumped_from_slice_indices=*/{}, "r"));
}

}  // namespace
}  // namespace xlscc
