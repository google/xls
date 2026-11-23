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

#include "xls/passes/tools/generate_documentation_proto.h"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/proto_test_utils.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/tools/pass_documentation.pb.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::Not;
using ::xls::proto_testing::EqualsProto;

class StubOptimizationPass final : public OptimizationPass {
 public:
  StubOptimizationPass(std::string_view short_name, std::string_view long_name)
      : OptimizationPass(short_name, long_name) {}

  RedundancyGuard GetRedundancyGuard(
      const OptimizationPassOptions& options,
      OptimizationContext& context) const override {
    return RedundancyGuard::CanSkip();
  }

 protected:
  absl::StatusOr<bool> RunInternal(
      Package* ir, const OptimizationPassOptions& options, PassResults* results,
      OptimizationContext& context) const override {
    return false;
  }
};

class StubPassGenerator final : public OptimizationPassGenerator {
 public:
  StubPassGenerator(const OptimizationPassRegistryBase& registry,
                    std::string_view short_name, std::string_view long_name)
      : OptimizationPassGenerator(registry),
        short_name_(short_name),
        long_name_(long_name) {}

  absl::StatusOr<std::unique_ptr<OptimizationPass>> Generate() const override {
    return std::make_unique<StubOptimizationPass>(short_name_, long_name_);
  }

  std::unique_ptr<OptimizationPassGenerator> Clone(
      const OptimizationPassRegistryBase& registry) const override {
    return std::make_unique<StubPassGenerator>(registry, short_name_,
                                               long_name_);
  }

 private:
  std::string short_name_;
  std::string long_name_;
};

void RegisterStubPass(OptimizationPassRegistry& registry,
                      std::string_view short_name, std::string_view long_name) {
  XLS_ASSERT_OK(registry.Register(
      short_name,
      std::make_unique<StubPassGenerator>(registry, short_name, long_name)));
}

constexpr std::string_view kFakeOptimizationPassHeader = R"cpp(
#ifndef FAKE_OPT_PASS_H_
#define FAKE_OPT_PASS_H_
  namespace xls {
  class OptimizationPass {};
  }  // namespace xls
#endif
)cpp";

TEST(GenerateDocumentationProtoTest, MultiHeaderXlsPassEndToEnd) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::filesystem::path md_path,
      GetXlsRunfilePath(
          "xls/passes/tools/"
          "passes_documentation.test_multi_header_documentation.md"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string markdown, GetFileContents(md_path));

  EXPECT_THAT(
      markdown,
      AllOf(HasSubstr("## test_multi_header - Test multi-header pass "
                      "{#test_multi_header}"),
            HasSubstr("Primary multi-header pass documentation."),
            HasSubstr("xls/passes/tools/test_multi_header_pass.h)"),
            Not(HasSubstr("Decoy base pass documentation in helper header.")),
            Not(HasSubstr("test_multi_header_helper.h")),
            Not(HasSubstr("Helper struct comment"))));
}

TEST(GenerateDocumentationProtoTest, EmptyHeadersReturnsEmptyProto) {
  OptimizationPassRegistry registry;
  RegisterStubPass(registry, "some_pass", "Some pass");
  registry.AddRegistrationInfo("some_pass", "SomePass", "xls/passes/some.h");

  EXPECT_THAT(GenerateDocumentationProto(
                  registry, absl::Span<PassHeaderPath const>(), /*copts=*/{}),
              IsOkAndHolds(EqualsProto(R"pb()pb")));
}

TEST(GenerateDocumentationProtoTest,
     ExactMatchInSecondHeaderBeatsDecoyBaseInFirstHeaderAndUsesRepoPath) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory temp_dir, TempDirectory::Create());
  XLS_ASSERT_OK(SetFileContents(temp_dir.path() / "fake_opt_pass.h",
                                kFakeOptimizationPassHeader));
  std::filesystem::path helper_disk_path = temp_dir.path() / "gen_helper.h";
  std::filesystem::path pass_disk_path = temp_dir.path() / "gen_my_pass.h";
  XLS_ASSERT_OK(SetFileContents(helper_disk_path, R"cpp(
#include "fake_opt_pass.h"
    namespace xls {
    // Decoy base pass comment in first header.
    class HelperBasePass : public OptimizationPass {};
    }  // namespace xls
  )cpp"));
  XLS_ASSERT_OK(SetFileContents(pass_disk_path, R"cpp(
#include "gen_helper.h"
    namespace xls {
    // Primary pass comment in second header.
    class MyPass : public HelperBasePass {};
    }  // namespace xls
  )cpp"));

  OptimizationPassRegistry registry;
  RegisterStubPass(registry, "my_pass", "My pass");
  const std::vector<std::string> registered_headers = {
      "xls/passes/gen_helper.h",
      "xls/passes/gen_my_pass.h",
  };
  registry.AddRegistrationInfo("my_pass", "xls::MyPass", registered_headers);

  const std::vector<PassHeaderPath> headers = {
      {.file_path = helper_disk_path.string(),
       .repo_path = "xls/passes/gen_helper.h"},
      {.file_path = pass_disk_path.string(),
       .repo_path = "xls/passes/gen_my_pass.h"},
  };
  std::vector<std::string> copts = {
      absl::StrCat("-I", temp_dir.path().string())};

  EXPECT_THAT(GenerateDocumentationProto(registry, headers, copts),
              IsOkAndHolds(EqualsProto(R"pb(
                passes {
                  short_name: "my_pass"
                  long_name: "My pass"
                  file: "xls/passes/gen_my_pass.h"
                  notes: "Primary pass comment in second header."
                }
              )pb")));
}

TEST(GenerateDocumentationProtoTest,
     SingleRegisteredHeaderFoundAmongMultipleTargetHeaders) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory temp_dir, TempDirectory::Create());
  XLS_ASSERT_OK(SetFileContents(temp_dir.path() / "fake_opt_pass.h",
                                kFakeOptimizationPassHeader));
  std::filesystem::path helper_disk_path = temp_dir.path() / "helper.h";
  std::filesystem::path pass_disk_path = temp_dir.path() / "cc_lib_pass.h";
  XLS_ASSERT_OK(SetFileContents(helper_disk_path, R"cpp(
    namespace xls {
    int HelperFn();
    }  // namespace xls
  )cpp"));
  XLS_ASSERT_OK(SetFileContents(pass_disk_path, R"cpp(
#include "fake_opt_pass.h"
    namespace xls {
    // Documentation for cc_library registered pass.
    class CcLibPass : public OptimizationPass {};
    }  // namespace xls
  )cpp"));

  OptimizationPassRegistry registry;
  RegisterStubPass(registry, "cc_lib_pass", "CC library pass");
  // Simulates REGISTER_OPT_PASS in cc_lib_pass.cc registering only
  // cc_lib_pass.h while the cc_library target exposes [helper.h,
  // cc_lib_pass.h].
  registry.AddRegistrationInfo("cc_lib_pass", "CcLibPass",
                               "xls/passes/cc_lib_pass.h");

  const std::vector<PassHeaderPath> headers = {
      {.file_path = helper_disk_path.string(),
       .repo_path = "xls/passes/helper.h"},
      {.file_path = pass_disk_path.string(),
       .repo_path = "xls/passes/cc_lib_pass.h"},
  };
  std::vector<std::string> copts = {
      absl::StrCat("-I", temp_dir.path().string())};

  EXPECT_THAT(GenerateDocumentationProto(registry, headers, copts),
              IsOkAndHolds(EqualsProto(R"pb(
                passes {
                  short_name: "cc_lib_pass"
                  long_name: "CC library pass"
                  file: "xls/passes/cc_lib_pass.h"
                  notes: "Documentation for cc_library registered pass."
                }
              )pb")));
}

TEST(GenerateDocumentationProtoTest,
     EmptyHelperCommentDoesNotShadowBasePassComment) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory temp_dir, TempDirectory::Create());
  XLS_ASSERT_OK(SetFileContents(temp_dir.path() / "fake_opt_pass.h",
                                kFakeOptimizationPassHeader));
  std::filesystem::path header = temp_dir.path() / "shadow_pass.h";
  XLS_ASSERT_OK(SetFileContents(header, R"cpp(
#include "fake_opt_pass.h"
    namespace xls {
    class UncommentedHelperPass : public OptimizationPass {};

    // Shared base pass documentation.
    class DocumentedBasePass : public OptimizationPass {};

    class ConcretePass : public DocumentedBasePass {};
    }  // namespace xls
  )cpp"));

  OptimizationPassRegistry registry;
  RegisterStubPass(registry, "concrete_pass", "Concrete pass");
  registry.AddRegistrationInfo("concrete_pass", "ConcretePass",
                               header.string());

  std::vector<std::string> copts = {
      absl::StrCat("-I", temp_dir.path().string())};
  std::string expected_proto = absl::StrCat(
      R"pb(
        passes { short_name: "concrete_pass" long_name: "Concrete pass"
                 notes: "Shared base pass documentation."
                 file: ")pb",
      header.string(), R"pb(" }
      )pb");
  EXPECT_THAT(GenerateDocumentationProto(registry, header.string(), copts),
              IsOkAndHolds(EqualsProto(std::string_view(expected_proto))));
}

TEST(GenerateDocumentationProtoTest,
     UncommentedDerivedPassFallsBackToBaseCommentKeepingDerivedHeaderPath) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory temp_dir, TempDirectory::Create());
  XLS_ASSERT_OK(SetFileContents(temp_dir.path() / "fake_opt_pass.h",
                                kFakeOptimizationPassHeader));
  std::filesystem::path base_disk_path = temp_dir.path() / "shared_base.h";
  std::filesystem::path derived_disk_path = temp_dir.path() / "derived_pass.h";
  XLS_ASSERT_OK(SetFileContents(base_disk_path, R"cpp(
#include "fake_opt_pass.h"
    namespace xls {
    // Shared ABC documentation in base header.
    class SharedBasePass : public OptimizationPass {};
    }  // namespace xls
  )cpp"));
  XLS_ASSERT_OK(SetFileContents(derived_disk_path, R"cpp(
#include "shared_base.h"
    namespace xls {
    class DerivedPass : public SharedBasePass {};
    }  // namespace xls
  )cpp"));

  OptimizationPassRegistry registry;
  RegisterStubPass(registry, "derived_pass", "Derived pass");
  const std::vector<std::string> registered_headers = {
      "xls/passes/shared_base.h",
      "xls/passes/derived_pass.h",
  };
  registry.AddRegistrationInfo("derived_pass", "DerivedPass",
                               registered_headers);

  const std::vector<PassHeaderPath> headers = {
      {.file_path = base_disk_path.string(),
       .repo_path = "xls/passes/shared_base.h"},
      {.file_path = derived_disk_path.string(),
       .repo_path = "xls/passes/derived_pass.h"},
  };
  std::vector<std::string> copts = {
      absl::StrCat("-I", temp_dir.path().string())};

  EXPECT_THAT(GenerateDocumentationProto(registry, headers, copts),
              IsOkAndHolds(EqualsProto(R"pb(
                passes {
                  short_name: "derived_pass"
                  long_name: "Derived pass"
                  file: "xls/passes/derived_pass.h"
                  notes: "Shared ABC documentation in base header."
                }
              )pb")));
}

}  // namespace
}  // namespace xls
