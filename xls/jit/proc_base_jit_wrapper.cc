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

#include "xls/jit/proc_base_jit_wrapper.h"

#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>

#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/jit/aot_entrypoint.pb.h"

namespace xls {

namespace {
// Static duration caches of packages and entry points that have jit wrappers.
// The memory cost of holding these is worth it to avoid creating a ton of
// copies.
static absl::Mutex CACHE_MUTEX(absl::kConstInit);
static absl::NoDestructor<
    absl::flat_hash_map<std::string_view, std::unique_ptr<Package>>>
    package_cache ABSL_GUARDED_BY(CACHE_MUTEX);
static absl::NoDestructor<absl::flat_hash_map<
    absl::Span<uint8_t const>, std::unique_ptr<AotPackageEntrypointsProto>>>
    entrypoints_cache ABSL_GUARDED_BY(CACHE_MUTEX);
}  // namespace

/* static */ absl::StatusOr<Package*> BaseProcJitWrapper::GetCachedPackage(
    std::string_view ir) {
  absl::MutexLock mu(CACHE_MUTEX);
  auto it = package_cache->find(ir);
  if (it == package_cache->end()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                         Parser::ParsePackage(ir));
    Package* ptr = package.get();
    package_cache->insert({ir, std::move(package)});
    return ptr;
  }
  return it->second.get();
}

/* static */ absl::StatusOr<AotPackageEntrypointsProto const*>
BaseProcJitWrapper::GetCachedEntrypoints(absl::Span<uint8_t const> pb) {
  absl::MutexLock mu(CACHE_MUTEX);
  auto it = entrypoints_cache->find(pb);
  if (it == entrypoints_cache->end()) {
    AotPackageEntrypointsProto proto;
    XLS_RET_CHECK(proto.ParseFromArray(pb.data(), pb.size()));
    return entrypoints_cache
        ->emplace(
            pb, std::make_unique<AotPackageEntrypointsProto>(std::move(proto)))
        .first->second.get();
  }
  return it->second.get();
}

}  // namespace xls
