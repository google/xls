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

#ifndef XLS_PASSES_PASS_REGISTRY_H_
#define XLS_PASSES_PASS_REGISTRY_H_

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Base class for a builder for generic passes.
template <typename OptionsT, typename... ContextT>
class PassGenerator {
 public:
  virtual ~PassGenerator() = default;
  // Create a new pass of the templated type
  virtual absl::StatusOr<std::unique_ptr<PassBase<OptionsT, ContextT...>>>
  Generate() const = 0;
};

// A registry for holding passes of a particular type. This allows one to
// request builders by name.
template <typename OptionsT, typename... ContextT>
class PassRegistry final {
 public:
  struct RegistrationInfo {
    std::string short_name;
    std::string class_name;
    std::string header_file;
    std::optional<std::string> comments;
  };
  using GeneratorPtr = std::unique_ptr<PassGenerator<OptionsT, ContextT...>>;
  constexpr PassRegistry() = default;
  constexpr ~PassRegistry() = default;

  // Register a generator with a given name.
  absl::Status Register(std::string_view name, GeneratorPtr gen) {
    absl::MutexLock mu(&registry_lock_);
    if (generators_.contains(name)) {
      return absl::AlreadyExistsError(
          absl::StrFormat("pass %s registered more than once", name));
    }
    generators_[name] = std::move(gen);
    return absl::OkStatus();
  }

  // Get a pass generator of the given name.
  absl::StatusOr<PassGenerator<OptionsT, ContextT...>*> Generator(
      std::string_view name) const {
    absl::MutexLock mu(&registry_lock_);
    if (!generators_.contains(name)) {
      return absl::NotFoundError(
          absl::StrFormat("%s is not registered. Have [%s]", name,
                          absl::StrJoin(GetRegisteredNames(), ", ")));
    }
    return generators_.at(name).get();
  }

  std::vector<std::string_view> GetRegisteredNames() const {
    absl::MutexLock mu(&registry_lock_);
    std::vector<std::string_view> res;
    res.reserve(generators_.size());
    for (const auto& [k, _] : generators_) {
      res.push_back(k);
    }
    absl::c_sort(res);
    return res;
  }

  std::vector<RegistrationInfo> GetRegisteredInfos() const {
    absl::MutexLock mu(&registry_lock_);
    std::vector<RegistrationInfo> res;
    res.reserve(registration_info_.size());
    for (const auto& [_, v] : registration_info_) {
      res.push_back(v);
    }
    for (const auto& [k, _] : generators_) {
      if (!registration_info_.contains(k)) {
        res.push_back({
            .short_name = std::string(k),
            .class_name = "UNKNOWN",
            .header_file = "UNKNOWN",
        });
      }
    }
    absl::c_sort(res, [](const RegistrationInfo& a, const RegistrationInfo& b) {
      return a.short_name < b.short_name;
    });
    return res;
  }

  void AddRegistrationInfo(std::string_view name, std::string_view class_name,
                           std::string_view header_file) {
    absl::MutexLock mu(&registry_lock_);
    registration_info_[name] = {
        .short_name = std::string(name),
        .class_name = std::string(class_name),
        .header_file = std::string(header_file),
    };
  }

 private:
  mutable absl::Mutex registry_lock_;
  absl::flat_hash_map<std::string, GeneratorPtr> generators_
      ABSL_GUARDED_BY(registry_lock_);
  absl::flat_hash_map<std::string, RegistrationInfo> registration_info_
      ABSL_GUARDED_BY(registry_lock_);
};

}  // namespace xls

#endif  // XLS_PASSES_PASS_REGISTRY_H_
