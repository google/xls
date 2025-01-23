# Releasing

## Versioning

The project use the following versioning scheme:
```
v${SEM_VER}-${COMMITS_COUNT_SINCE_LAST_ANNOTATED_TAG}-g${GIT_HASH)
```

> The [v0.0.0](https://github.com/google/xls/tree/v0.0.0) annotated tag points to the initial commit to bootstrap the versioning scheme using [git-describe](https://git-scm.com/docs/git-describe).

Released binaries are stamped with the `{version}` using [`--embed_label`](https://bazel.build/reference/command-line-reference#:~:text=%2D%2Dembed_label%3D%3Ca%20one,label%20in%20binary) during `bazel build` invocations.

Soft-tags (non-annotated) are used to mark released versions.

## Packaging

The `//dist:xls_dist_tar` target uses [@rules_pkg](https://github.com/bazelbuild/rules_pkg/)/pkg_tar to create an archive named `xls-{version}-{os}-{arch}.tar.gz` with the following layout:

```
xls-{version}-{os}-{arch}/codegen_main
xls-{version}-{os}-{arch}/interpreter_main
xls-{version}-{os}-{arch}/ir_converter_main
xls-{version}-{os}-{arch}/opt_main
xls-{version}-{os}-{arch}/proto_to_dslx_main
...
xls-{version}-{os}-{arch}/LICENSE
xls-{version}-{os}-{arch}/THIRD_PARTY_NOTICES.txt
```

`{version}`, `{os}` and `{arch}` placeholders get replaced by [user-defined build settings](https://bazel.build/extending/config#user-defined-build-settings) passed during `bazel build` invocations.

## Automation

The [Nightly Ubuntu 22.04](https://github.com/google/xls/actions/workflows/nightly-ubuntu-22.04.yml) workflow automates the release process.

The `Dist` step builds the `//dist:xls_dist_tar` target w/ the following flags:

| flag               | value                     |
| ------------------ | ------------------------- |
| `--embed_label`    | `$VERSION`                |
| `--//dist:version` | `$VERSION`                |
| `--//dist:os`      | `downcased($RUNNER_OS)`   |
| `--//dist:arch`    | `downcased($RUNNER_ARCH)` |

The `Release` step uploads the release artefacts to https://github.com/google/xls/releases using [softprops/action-gh-release@v1](https://github.com/marketplace/actions/gh-release) and creates new soft-tag corresponding to `{version}`.
