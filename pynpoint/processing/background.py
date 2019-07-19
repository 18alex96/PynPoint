"""
Pipeline modules for subtraction of the background emission.
"""

import sys
import time
import warnings

import numpy as np

from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.image import create_mask
from pynpoint.util.module import progress
from pynpoint.util.timestamps import create_timestamp_list, calc_sky_frame


class SimpleBackgroundSubtractionModule(ProcessingModule):
    """
    Pipeline module for simple background subtraction. Only applicable on data obtained with
    dithering.
    """

    __author__ = 'Markus Bonse, Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 shift: int) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output.
        shift : int
            Frame index offset for the background subtraction. Typically equal to the number of
            frames per dither location.

        Returns
        -------
        NoneType
            None
        """

        super(SimpleBackgroundSubtractionModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_shift = shift

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Simple background subtraction with a constant index offset.

        Returns
        -------
        NoneType
            None
        """

        nframes = self.m_image_in_port.get_shape()[0]

        subtract = self.m_image_in_port[0] - self.m_image_in_port[(0 + self.m_shift) % nframes]

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            self.m_image_out_port[0] = subtract
        else:
            self.m_image_out_port.set_all(subtract, data_dim=3)
        start_time = time.time()
        for i in range(1, nframes):
            progress(i, nframes, 'Running SimpleBackgroundSubtractionModule...', start_time)

            subtract = self.m_image_in_port[i] - self.m_image_in_port[(i + self.m_shift) % nframes]

            if self.m_image_in_port.tag == self.m_image_out_port.tag:
                self.m_image_out_port[i] = subtract
            else:
                self.m_image_out_port.append(subtract)

        sys.stdout.write('Running SimpleBackgroundSubtractionModule... [DONE]\n')
        sys.stdout.flush()

        history = f'shift = {self.m_shift}'
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('SimpleBackgroundSubtractionModule', history)
        self.m_image_out_port.close_port()


class MeanBackgroundSubtractionModule(ProcessingModule):
    """
    Pipeline module for mean background subtraction. Only applicable on data obtained with
    dithering.
    """

    __author__ = 'Markus Bonse, Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 shift: int = None,
                 cubes: int = 1) -> None:
        """
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output. Should be different from
            *image_in_tag*.
        shift : int, None
            Image index offset for the background subtraction. Typically equal to the number of
            frames per dither location. If set to None, the ``NFRAMES`` attribute will be used to
            select the background frames automatically. The *cubes* parameters should be set when
            *shift* is set to None.
        cubes : int
            Number of consecutive cubes per dithering position.

        Returns
        -------
        NoneType
            None
        """

        super(MeanBackgroundSubtractionModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_shift = shift
        self.m_cubes = cubes

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Mean background subtraction which uses either a constant index
        offset or the ``NFRAMES`` attributes. The mean background is calculated from the cubes
        before and after the science cube.

        Returns
        -------
        NoneType
            None
        """

        # Use NFRAMES values if shift=None
        if self.m_shift is None:
            self.m_shift = self.m_image_in_port.get_attribute('NFRAMES')

        nframes = self.m_image_in_port.get_shape()[0]

        if not isinstance(self.m_shift, np.ndarray) and nframes < self.m_shift*2.0:
            raise ValueError('The input stack is too small for a mean background subtraction. The '
                             'position of the star should shift at least once.')

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            raise ValueError('The tag of the input port should be different from the output port.')

        # Number of substacks
        if isinstance(self.m_shift, np.ndarray):
            nstacks = np.size(self.m_shift)
        else:
            nstacks = int(np.floor(nframes/self.m_shift))

        # First mean subtraction to set up the output port array
        if isinstance(self.m_shift, np.ndarray):
            next_start = np.sum(self.m_shift[0:self.m_cubes])
            next_end = np.sum(self.m_shift[0:2*self.m_cubes])

            if 2*self.m_cubes > np.size(self.m_shift):
                raise ValueError('Not enough frames available for the background subtraction.')

            bg_data = self.m_image_in_port[next_start:next_end, ]
            bg_mean = np.mean(bg_data, axis=0)

        else:
            bg_data = self.m_image_in_port[self.m_shift:2*self.m_shift, ]
            bg_mean = np.mean(bg_data, axis=0)

        # Initiate the result port data with the first frame
        bg_sub = self.m_image_in_port[0, ] - bg_mean
        self.m_image_out_port.set_all(bg_sub, data_dim=3)

        # Mean subtraction of the first stack (minus the first frame)
        if isinstance(self.m_shift, np.ndarray):
            tmp_data = self.m_image_in_port[1:next_start, ]
            tmp_data = tmp_data - bg_mean
            self.m_image_out_port.append(tmp_data)

        else:
            tmp_data = self.m_image_in_port[1:self.m_shift, ]
            tmp_data = tmp_data - bg_mean
            self.m_image_out_port.append(tmp_data)

        # Processing of the rest of the data
        start_time = time.time()
        if isinstance(self.m_shift, np.ndarray):
            for i in range(self.m_cubes, nstacks, self.m_cubes):
                progress(i, nstacks, 'Running MeanBackgroundSubtractionModule...', start_time)

                prev_start = np.sum(self.m_shift[0:i-self.m_cubes])
                prev_end = np.sum(self.m_shift[0:i])

                next_start = np.sum(self.m_shift[0:i+self.m_cubes])
                next_end = np.sum(self.m_shift[0:i+2*self.m_cubes])

                # calc the mean (previous)
                tmp_data = self.m_image_in_port[prev_start:prev_end, ]
                tmp_mean = np.mean(tmp_data, axis=0)

                if i < nstacks-self.m_cubes:
                    # calc the mean (next)
                    tmp_data = self.m_image_in_port[next_start:next_end, ]
                    tmp_mean = (tmp_mean + np.mean(tmp_data, axis=0)) / 2.0

                # subtract mean
                tmp_data = self.m_image_in_port[prev_end:next_start, ]
                tmp_data = tmp_data - tmp_mean
                self.m_image_out_port.append(tmp_data)

        else:
            # the last and the one before will be performed afterwards
            top = int(np.ceil(nframes/self.m_shift)) - 2

            for i in range(1, top, 1):
                progress(i, top, 'Running MeanBackgroundSubtractionModule...', start_time)

                # calc the mean (next)
                tmp_data = self.m_image_in_port[(i+1)*self.m_shift:(i+2)*self.m_shift, ]
                tmp_mean = np.mean(tmp_data, axis=0)

                # calc the mean (previous)
                tmp_data = self.m_image_in_port[(i-1)*self.m_shift:(i+0)*self.m_shift, ]
                tmp_mean = (tmp_mean + np.mean(tmp_data, axis=0)) / 2.0

                # subtract mean
                tmp_data = self.m_image_in_port[(i+0)*self.m_shift:(i+1)*self.m_shift, ]
                tmp_data = tmp_data - tmp_mean
                self.m_image_out_port.append(tmp_data)

            # last and the one before
            # 1. ------------------------------- one before -------------------
            # calc the mean (previous)
            tmp_data = self.m_image_in_port[(top-1)*self.m_shift:(top+0)*self.m_shift, ]
            tmp_mean = np.mean(tmp_data, axis=0)

            # calc the mean (next)
            # 'nframes' is important if the last step is to huge
            tmp_data = self.m_image_in_port[(top+1)*self.m_shift:nframes, ]
            tmp_mean = (tmp_mean + np.mean(tmp_data, axis=0)) / 2.0

            # subtract mean
            tmp_data = self.m_image_in_port[top*self.m_shift:(top+1)*self.m_shift, ]
            tmp_data = tmp_data - tmp_mean
            self.m_image_out_port.append(tmp_data)

            # 2. ------------------------------- last -------------------
            # calc the mean (previous)
            tmp_data = self.m_image_in_port[(top+0)*self.m_shift:(top+1)*self.m_shift, ]
            tmp_mean = np.mean(tmp_data, axis=0)

            # subtract mean
            tmp_data = self.m_image_in_port[(top+1)*self.m_shift:nframes, ]
            tmp_data = tmp_data - tmp_mean
            self.m_image_out_port.append(tmp_data)
            # -----------------------------------------------------------

        sys.stdout.write('Running MeanBackgroundSubtractionModule... [DONE]\n')
        sys.stdout.flush()

        if isinstance(self.m_shift, np.ndarray):
            history = f'shift = NFRAMES'
        else:
            history = f'shift = {self.m_shift}'

        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('MeanBackgroundSubtractionModule', history)
        self.m_image_out_port.close_port()


class LineSubtractionModule(ProcessingModule):
    """
    Pipeline module for subtracting the background emission from each pixel by computing the mean
    or median of all values in the row and column of the pixel. The module can for example be
    used if no background data is available or to remove a detector bias.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 combine: str = 'median',
                 mask=None) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output.
        combine : str
            The method by which the column and row pixel values are combined ('median' or 'mean').
            Using a mean-combination is computationally faster than a median-combination.
        mask : float, None
            The radius of the mask within which pixel values are ignored. No mask is used if set
            to None.

        Returns
        -------
        NoneType
            None
        """

        super(LineSubtractionModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_combine = combine
        self.m_mask = mask

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Selects the pixel values in the column and row at each pixel
        position, computes the mean or median value while excluding pixels within the radius of
        the mask, and subtracts the mean or median value from each pixel separately.

        Returns
        -------
        NoneType
            None
        """

        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
        im_shape = self.m_image_in_port.get_shape()[-2:]

        def _subtract_line(image_in, mask):
            image_tmp = np.copy(image_in)
            image_tmp[mask == 0.] = np.nan

            if self.m_combine == 'mean':
                row_mean = np.nanmean(image_tmp, axis=1)
                col_mean = np.nanmean(image_tmp, axis=0)

                x_grid, y_grid = np.meshgrid(col_mean, row_mean)
                subtract = (x_grid+y_grid)/2.

            elif self.m_combine == 'median':
                subtract = np.zeros(im_shape)

                col_median = np.nanmedian(image_tmp, axis=0)
                col_2d = np.tile(col_median, (im_shape[1], 1))

                image_tmp -= col_2d
                image_tmp[mask == 0.] = np.nan

                row_median = np.nanmedian(image_tmp, axis=1)
                row_2d = np.tile(row_median, (im_shape[0], 1))
                row_2d = np.rot90(row_2d)  # 90 deg rotation in clockwise direction

                subtract = col_2d + row_2d

            return image_in - subtract

        if self.m_mask:
            size = (self.m_mask/pixscale, None)
        else:
            size = (None, None)

        mask = create_mask(im_shape, size)

        self.apply_function_to_images(_subtract_line,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      'Running LineSubtractionModule',
                                      func_args=(mask, ))

        history = f'combine = {self.m_combine}'
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('LineSubtractionModule', history)
        self.m_image_out_port.close_port()


class NoddingBackgroundModule(ProcessingModule):
    """
    Pipeline module for background subtraction of data obtained with nodding (e.g., NACO AGPM
    data). Before using this module, the sky images should be stacked with the StackCubesModule
    such that each image in the stack of sky images corresponds to the mean combination of a
    single FITS data cube.
    """

    __author__ = 'Markus Bonse, Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 science_in_tag: str,
                 sky_in_tag: str,
                 image_out_tag: str,
                 mode: str = 'both') -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        science_in_tag : str
            Tag of the database entry with science images that are read as input.
        sky_in_tag : str
            Tag of the database entry with sky images that are read as input. The
            :class:`~pynpoint.processing.stacksubset.StackCubesModule` should be used on the sky
            images beforehand.
        image_out_tag : str
            Tag of the database entry with sky subtracted images that are written as output.
        mode : str
            Sky images that are subtracted, relative to the science images. Either the next,
            previous, or average of the next and previous cubes of sky frames can be used by
            choosing 'next', 'previous', or 'both', respectively.

        Returns
        -------
        NoneType
            None
        """

        super(NoddingBackgroundModule, self).__init__(name_in=name_in)

        self.m_science_in_port = self.add_input_port(science_in_tag)
        self.m_sky_in_port = self.add_input_port(sky_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        if mode in ['next', 'previous', 'both']:
            self.m_mode = mode
        else:
            raise ValueError('Mode needs to be \'next\', \'previous\', or \'both\'.')

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Create list of time stamps, get sky and science images, and
        subtract the sky images from the science images.

        Returns
        -------
        NoneType
            None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        time_stamps = create_timestamp_list(input_port_1=self.m_science_in_port,
                                            input_port_2=self.m_sky_in_port,
                                            type_1="SCIENCE",
                                            type_2="SKY")

        start_time = time.time()
        for i, time_entry in enumerate(time_stamps):
            progress(i, len(time_stamps), 'Running NoddingBackgroundModule...', start_time)

            if time_entry.m_im_type == 'SKY':
                continue

            sky = calc_sky_frame(time_stamps=time_stamps,
                                 sky_in_port=self.m_sky_in_port,
                                 index_of_science_data=i,
                                 mode=self.m_mode)
            science = self.m_science_in_port[time_entry.m_index, ]

            self.m_image_out_port.append(science - sky[None, ], data_dim=3)

        sys.stdout.write('Running NoddingBackgroundModule... [DONE]\n')
        sys.stdout.flush()

        history = f'mode = {self.m_mode}'
        self.m_image_out_port.copy_attributes(self.m_science_in_port)
        self.m_image_out_port.add_history('NoddingBackgroundModule', history)
        self.m_image_out_port.close_port()
