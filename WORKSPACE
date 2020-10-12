workspace(name = "com_google_xls")

load("//dependency_support/systemlibs:syslibs_configure.bzl", "syslibs_configure")
syslibs_configure(name = "local_config_syslibs")
load("@local_config_syslibs//:build_defs.bzl", "SYSTEM_LIBS_LIST")

print("Enabled external system libs: %s" % SYSTEM_LIBS_LIST)

# Load and configure a hermetic LLVM based C/C++ toolchain. This is done here
# and not in load_external.bzl because it requires several sequential steps of
# declaring archives and using things in them, which is awkward to do in .bzl
# files because it's not allowed to use `load` inside of a function.
load("//dependency_support:repo.bzl", "xls_http_archive")
xls_http_archive(
    name = "com_grail_bazel_toolchain",
    urls = [
        "https://github.com/grailbio/bazel-toolchain/archive/f4c17a3ae40f927ff62cc0fb8fe22b1530871807.zip",
    ],
    strip_prefix = "bazel-toolchain-f4c17a3ae40f927ff62cc0fb8fe22b1530871807",
    sha256 = "715fd98d566ed1304cb53e0c640427cf0916ec6db89588e3ac2b6a87632276d4",
    patches = ["//dependency_support/com_grail_bazel_toolchain:google_workstation_workaround.patch"],
    system_build_file = "//dependency_support/com_grail_bazel_toolchain:BUILD",
    system_link_files = {
        "//dependency_support/systemlibs:com_grail_bazel_toolchain.toolchain.BUILD": "toolchain/BUILD",
        "//dependency_support/systemlibs:com_grail_bazel_toolchain.toolchain.deps.bzl": "toolchain/deps.bzl",
        "//dependency_support/systemlibs:com_grail_bazel_toolchain.toolchain.rules.bzl": "toolchain/rules.bzl",
        "//dependency_support/systemlibs:com_grail_bazel_toolchain.internal.BUILD": "toolchain/internal/BUILD",
        "//dependency_support/systemlibs:com_grail_bazel_toolchain.internal.configure.bzl": "toolchain/internal/configure.bzl",
    },
)
load("@com_grail_bazel_toolchain//toolchain:deps.bzl", "bazel_toolchain_dependencies")
bazel_toolchain_dependencies()
load("@com_grail_bazel_toolchain//toolchain:rules.bzl", "llvm_toolchain")
llvm_toolchain(
    name = "llvm_toolchain",
    llvm_version = "10.0.0",
)
load("@llvm_toolchain//:toolchains.bzl", "llvm_register_toolchains")
llvm_register_toolchains()

load("//dependency_support:load_external.bzl", "load_external_repositories")
load_external_repositories()

# gRPC deps should be loaded before initializing other repos. Otherwise, various
# errors occur during repo loading and initialization.
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()

load("//dependency_support:initialize_external.bzl", "initialize_external_repositories")
initialize_external_repositories()

# Loading the extra deps must be called after initialize_eternal_repositories or
# the call to pip_install fails.
load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
grpc_extra_deps()

load("@xls_pip_deps//:requirements.bzl", "pip_install")
pip_install()
