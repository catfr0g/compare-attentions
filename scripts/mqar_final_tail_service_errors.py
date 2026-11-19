"""Record actual service exception metadata before inherited suppression wraps it."""

from pathlib import Path

from mqar_final_tail_errors import persist


def instrument(service, work):
    """Wrap only this owned service instance, never a shared SDK or scientific method."""
    invoke = service.invoke
    path = Path(work) / "service-failure.json"

    def observed(phase, operation):
        def boundary():
            try:
                return operation()
            except Exception as error:
                try:
                    if not path.exists():
                        persist(path, error, phase)
                except Exception:  # Diagnostic I/O must not replace the original service failure.
                    pass
                raise

        return invoke(phase, boundary)

    service.invoke = observed
    return service
