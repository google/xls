# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

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

## Pull Request Style

We ask contributors to squash all the commits in the PR into a single one, in
order to have a cleaner revision history.

Generally, this can be accomplished by:

```
git merge-base main my-branch-name  # Tells you common ancesor COMMIT_HASH.
git reset --soft $COMMIT_HASH
git commit -a -m "My awesome squashed commit message!!!1"
```

See also [this Stack Overflow
question](https://stackoverflow.com/questions/17354353/git-squash-all-commits-in-branch-without-conflicting).

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).
