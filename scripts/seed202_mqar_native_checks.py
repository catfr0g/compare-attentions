"""Explicit view of the authenticated AST-loaded native predicate functions."""


class NativeChecks:
    """Expose the existing function objects, without replacing or relaxing predicates."""

    def __init__(self, namespace):
        self.finite_full_state = namespace["finite_full_state"]
        self.target_runtime = namespace["target_runtime"]
        self.compiler_evidence = namespace["compiler_evidence"]


def native_checks():
    from quality_inherit import native_checks as authenticated_checks

    return NativeChecks(vars(authenticated_checks()))
