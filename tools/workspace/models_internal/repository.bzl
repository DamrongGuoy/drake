# -*- python -*-

load("@drake//tools/workspace:github.bzl", "github_archive")

def models_internal_repository(
        name,
        mirrors = None):
    github_archive(
        name = name,
        repository = "RobotLocomotion/models",
        # XXX Temporary git sha from RobotLocomotion/models/pull/19.
        commit = "72854ca8a04e98670de6821e078fa9baf1fbbd91",
        sha256 = "c7f5a7319658b41bb33ee6676f7e70abd141c85da0860e953d65094af463a991",  # noqa
        build_file = ":package.BUILD.bazel",
        mirrors = mirrors,
    )
