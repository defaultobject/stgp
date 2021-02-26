"""decorators used in gpajx."""
import inspect
from .errors import StrictModeError
from . import settings


def strict_mode_check(func):
    """Enforce strict model if required.

    When in strict mode, if funchas optional params that are not passed then
        this decorator will raise a RunTimeError
    """
    # Retrieve the original argspec
    num_args = len(getattr(func, "__argspec", inspect.getargspec(func).args))

    def inner(*args, **kwargs):

        if settings.in_strict_mode:
            if len(args) + len(kwargs.keys()) < num_args:
                # there must be a default argument being used
                raise StrictModeError()

        return func(*args, **kwargs)

    return inner
