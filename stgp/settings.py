"""Global settings and setters/getters for gpjax."""

import bibtexparser
import pathlib

in_strict_mode = False

use_loop_mode = False

force_black_box = False

safe_mode = False

use_quadrature = False


jitter = 1e-5
ng_jitter = 1e-7
ng_samples = 10


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

class use_loops:
    """Use loops instead of batching.

    Use case:
        with settings.use_loops():
            ...
    """

    def __init__(self, state=True, num_probe_vectors=1):
        """Store strict_mode state before with statement."""
        global use_loop_mode
        self.orig_value = use_loop_mode

    def __enter__(self):
        """Set strict_mode state to true."""
        global use_loop_mode
        use_loop_mode = True

    def __exit__(self, *args):
        """Restore strict_mode state."""
        global use_loop_mode
        use_loop_mode = self.orig_value


# global list of who to cite for the current model created
to_cite = []


def add_citation(arg):
    """Append citation to global to_cite list."""
    global to_cite
    to_cite.append(arg)


def print_citations():
    """Print bibtex citations."""
    global to_cite

    # this files path
    root = pathlib.Path(__file__).parent.absolute()

    # references are stored within gpjax, so use the root to this file to load
    with open(f"{root}/references/references.bib") as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)

    bib_entries = bib_database.entries

    # TODO: print print these somehow
    for _id in to_cite:
        for entry in bib_entries:
            if _id == entry["ID"]:
                print(entry)
