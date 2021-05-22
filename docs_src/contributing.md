# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement (CLA). You (or your employer) retain the copyright to your
contribution; this simply gives us permission to use and redistribute your
contributions as part of the project. Head over to
<https://cla.developers.google.com/> to see your current agreements on file or
to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code style

When writing code contributions to the project, please make sure to follow the
style guides:
The [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
and the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
There are a few small [XLS clarifications](xls_style.md) for local
style on this project where the style guide is ambiguous.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

### Pull Request Style

We ask contributors to squash all the commits in the PR into a single one, in
order to have a cleaner revision history.

Generally, this can be accomplished by:

```console
proj/xls$ # Here we assume origin points to google/xls.
proj/xls$ git fetch origin main
proj/xls$ git merge-base origin/main my-branch-name  # Tells you common ancesor COMMIT_HASH.
proj/xls$ git reset --soft $COMMIT_HASH
proj/xls$ git commit -a -m "My awesome squashed commit message!!!1"
proj/xls$ # Now we can more easily rebase our squashed commit on main.
proj/xls$ git rebase origin/main
```

Rebased branches can be pushed to their corresponding PRs with `--force`.

See also [this Stack Overflow
question](https://stackoverflow.com/questions/17354353/git-squash-all-commits-in-branch-without-conflicting).

## Rendering Documentation

XLS uses [mkdocs](https://www.mkdocs.org/) to render its documentation, and
serves it via GitHub pages at <https://google.github.io/xls>. To render
documentation locally as a preview, you can set up mkdocs as follows:

```console
proj/xls$ mkvirtualenv xls-mkdocs-env
proj/xls$ pip install mkdocs-material mkdocs-exclude mdx_truly_sane_lists
proj/xls$ mkdocs serve
```

This will start a local server that you can browse to and that will update the
documentation on the fly as you make changes.

Note that the `mkvirtualenv` command assumes you're using
[virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/index.html)
to manage your Python environment. You'll need to adjust these instrutions if
you're doing something different. That can include explicitly adding `mkdocs` to
your path, if locally installed Python binaries aren't available by default.

### DSL snippets in documentation

There are a few different language annotations we use in different
circumstances in the Markdown docs:

* `dslx`: A full code block that should be parsed/typechecked/tested.
* `dslx-snippet`: A fragment that should be syntax highlighted, but not
  parsed/typechecked/tested.
* `dslx-bad`: An example of something that we expect to produce an error
  when parsing/typechecking/testing.

GitHub issue [google/xls#378](https://github.com/google/xls/issues/378) tracks
a script that does the parse/typecheck/test that ensures our documentation is
up to date and correct.
