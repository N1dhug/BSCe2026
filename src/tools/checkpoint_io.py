"""Bounded retries for Windows sharing violations in atomic report I/O."""

import os
from time import sleep


_RETRY_DELAYS = (0.1, 0.2, 0.4, 0.8, 1.6)


def _sharing_violation(error):
    return getattr(error, 'winerror', None) in (32, 33)


def _retry(operation, *args, **kwargs):
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            return operation(*args, **kwargs)
        except OSError as error:
            if not _sharing_violation(error) or attempt == len(_RETRY_DELAYS):
                raise
            sleep(_RETRY_DELAYS[attempt])


def replace_file(source, destination):
    """Replace only with a fully written file; retain atomic rename semantics."""
    return _retry(os.replace, source, destination)


def discard_file(path):
    """Remove a redundant file, or defer cleanup if Windows still holds it.

    Call only for a closed temporary file or a complete checkpoint whose final
    raw unit has already been published. Never discard a partial checkpoint.
    Other filesystem errors propagate, and failed publication is never accepted.
    """
    try:
        _retry(path.unlink, missing_ok=True)
    except OSError as error:
        if not _sharing_violation(error):
            raise
        print(f'Cleanup deferred for {path}; saved data retained.', flush=True)
        return False
    return True
