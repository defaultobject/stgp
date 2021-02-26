in_strict_mode = False

class strict_mode():
    def __init__(self, state=True, num_probe_vectors=1):
        global in_strict_mode
        self.orig_value = in_strict_mode

    def __enter__(self):
        global in_strict_mode
        in_strict_mode = True

    def __exit__(self, *args):
        global in_strict_mode
        in_strict_mode = self.orig_value


