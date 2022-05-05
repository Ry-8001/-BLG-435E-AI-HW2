def linear_annealing(init_value, min_value, decay_range):
    """ Decay and return the value at every call linearly.
        Arguments:
            - init_value: Initial value
            - min_value: Minimum value
            - decay_range: Range of the decaying process in terms of
            iterations.
    """
    value = init_value
    decay = (init_value - min_value) / decay_range

    while True:
        yield value
        value = max(min_value, value - decay)


def exponential_annealing(init_value, min_value, decay_ratio):
    """ Decay and return the value at every call exponentially.
        Arguments:
            - init_value: Initial value
            - final_value: Minimum value
            - decay_ratio: Decaying rate
    """
    value = init_value

    while True:
        yield value
        value = max(min_value, value*decay_ratio)
