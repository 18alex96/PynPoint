"""
Functions for creating and working with timestamps.
"""
import warnings

import numpy as np
from typeguard import typechecked

from pynpoint.core.dataio import InputPort


class TimeStamp:
    """
    Class for creating a time stamp.
    """

    def __init__(self, time_in, im_type, index):
        self.m_time = time_in
        self.m_im_type = im_type
        self.m_index = index

    def __repr__(self):
        return repr((self.m_time,
                     self.m_im_type,
                     self.m_index))


@typechecked
def create_timestamp_list(input_port_1: InputPort,
                          input_port_2: InputPort,
                          type_1: str = "SCIENCE",
                          type_2: str = "SKY") -> list:
    """
    Method for assigning timestamps, based on the exposure number ID, to each individual
    frame of two corresponding data tags (science & sky or science & center).

    Parameters
    ----------
    input_port_1 : First input port that is used to create the times stamps.
    input_port_2 : Second input port that is used to create the times stamps. Should be
        a port pointing towrads SKY or CENTER frames.

    Returns
    -------
    tuple(int, int)
        Pixel position (y, x) of the image center.
    """

    if type_2 not in ('SKY', 'CENTER'):
        raise ValueError(f"The *type_2* argument should be either SKY or CENTER.")

    exp_no_1 = input_port_1.get_attribute('EXP_NO')
    exp_no_2 = input_port_2.get_attribute('EXP_NO')

    nframes_1 = input_port_1.get_attribute('NFRAMES')
    nframes_2 = input_port_2.get_attribute('NFRAMES')

    if np.all(nframes_2 != 1):
        warnings.warn(f'The NFRAMES values of {input_port_2.tag} are not all equal to unity. '
                      'The StackCubesModule should be applied on these images before this '
                      'module is used.')

    time_stamps = []

    for i, item in enumerate(exp_no_2):
        time_stamps.append(TimeStamp(item, type_2, i))

    current = 0
    for i, item in enumerate(exp_no_1):
        frames = slice(current, current + nframes_1[i])
        time_stamps.append(TimeStamp(item, type_1, frames))
        current += nframes_1[i]

    time_stamps = sorted(time_stamps, key=lambda time_stamp: time_stamp.m_time)

    return time_stamps


@typechecked
def calc_sky_frame(time_stamps: list,
                   sky_in_port: InputPort,
                   index_of_science_data: int,
                   mode: str) -> np.ndarray:
    """
    Method for finding the required sky frame (next, previous, or the mean of next and
    previous) by comparing the time stamp of the science frame with preceding and following
    sky frames.
    """

    if not any(x.m_im_type == 'SKY' for x in time_stamps):
        raise ValueError('List of time stamps does not contain any SKY images.')

    def search_for_next_sky():
        for i in range(index_of_science_data, len(time_stamps)):
            if time_stamps[i].m_im_type == 'SKY':
                return sky_in_port[time_stamps[i].m_index,]

        # no next sky found, look for previous sky
        return search_for_previous_sky()

    def search_for_previous_sky():
        for i in reversed(list(range(0, index_of_science_data))):
            if time_stamps[i].m_im_type == 'SKY':
                return sky_in_port[time_stamps[i].m_index,]

        # no previous sky found, look for next sky
        return search_for_next_sky()

    if mode == 'next':
        return search_for_next_sky()

    if mode == 'previous':
        return search_for_previous_sky()

    if mode == 'both':
        previous_sky = search_for_previous_sky()
        next_sky = search_for_next_sky()

        return (previous_sky + next_sky) / 2.
