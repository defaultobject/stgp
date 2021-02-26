"""Global settings and setters/getters for gpjax."""
in_strict_mode = False


class strict_mode:
    """Enable strict_mode.

    Use case:
        with settings.strict_mode():
            ...
    """

    def __init__(self, state=True, num_probe_vectors=1):
        """Store strict_mode state before with statement."""
        global in_strict_mode
        self.orig_value = in_strict_mode

    def __enter__(self):
        """Set strict_mode state to true."""
        global in_strict_mode
        in_strict_mode = True

    def __exit__(self, *args):
        """Restore strict_mode state."""
        global in_strict_mode
        in_strict_mode = self.orig_value


# global list of who to cite for the current model created
to_cite = []


def add_citation(arg):
    """Append citation to global to_cite list."""
    global to_cite
    to_cite.append(arg)
