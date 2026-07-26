from collections import namedtuple

def make_transition_class(labels="state action reward next_state done"):
    """
    Factory that builds and returns a namedtuple *class* customized
    with the provided labels. Call this ONCE at setup time, then use
    the returned class to build instances per-step.
    """
    if isinstance(labels, str):
        labels = labels.replace(',', ' ').split()

    return namedtuple('Transition', labels)