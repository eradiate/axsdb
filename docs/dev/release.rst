Release guide
=============

The release of AxsDB to PyPI is automated with GitHub Actions (see the
`release.yml` workflow). It is triggered when a tag is pushed to the repository.
The steps for a release are as follows:

1. Check the release notes (``CHANGELOG.md``) and make sure that the list of
   changes, target version number and release date are correct.
2. Bump the ``version`` field in ``pyproject.toml`` to the target version
   number. The release workflow will fail if the version number and tag do not
   align. Push to GitHub. Note that the released commit does not have to be on
   the ``main`` branch (this is useful when working on release candidates, which
   can be deleted once everything works).
3. Create a version tag ``vX.Y.Z``, either locally (you will have to push it),
   or on GitHub using the
   `Release page <https://github.com/eradiate/axsdb/releases>`__.
4. Monitor the publication.
5. When publication is successful, bump the version to the next dev patch
   ``vX.Y.(Z+1).dev0``, commit and push.
