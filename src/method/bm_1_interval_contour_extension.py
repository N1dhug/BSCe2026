"""Explicit, opt-in enlargement of the H2 terminal contour bank.

The original five-center prefix and original native implementation are
preserved. Successful original updates exit before any added center, so
their arithmetic and return payload are unchanged. This adapter is not
installed into any historical or production runner by importing this file.
"""

ORIGINAL_CENTERS = (.5, .25, .75, 0., 1.)
EXTENDED_CENTERS = ORIGINAL_CENTERS + (.125, .375, .625, .875)


def native_terminal_update_extended(operator, logs, value, sigma_y, settings, *, mapper=None):
    """Use the declared nine-center bank with every existing guard intact.

This is a finite representation change, not a guarantee that every possible
candidate will resolve. Original failures and source/job identities must be
preserved by any future provenance-aware adoption adapter.
    """
    from src.method.bm_1_interval_native import native_terminal_update
    if tuple(settings['centers']) != ORIGINAL_CENTERS:
        raise ValueError('Require the exact original ordered contour prefix.')
    extended = dict(settings, centers=list(EXTENDED_CENTERS))
    return native_terminal_update(operator, logs, value, sigma_y, extended, mapper=mapper)
