# Copyright (c) 2017, John Skinner
import enum


class TrackingState(enum.Enum):
    NOT_INITIALIZED = 0
    OK = 1
    LOST = 2


def tracking_state_from_string(s_state):
    if s_state == str(TrackingState.OK):
        return TrackingState.OK
    elif s_state == str(TrackingState.NOT_INITIALIZED):
        return TrackingState.NOT_INITIALIZED
    else:
        return TrackingState.LOST
