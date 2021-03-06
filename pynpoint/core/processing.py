"""
Interfaces for pipeline modules.
"""

from __future__ import absolute_import

import os
import sys
import warnings

from abc import abstractmethod, ABCMeta

import six
import numpy as np

from pynpoint.core.dataio import ConfigPort, InputPort, OutputPort
from pynpoint.util.multiproc import LineProcessingCapsule, apply_function
from pynpoint.util.module import progress, memory_frames


class PypelineModule(six.with_metaclass(ABCMeta)):
    """
    Abstract interface for the PypelineModule:

        * Reading Module (:class:`pynpoint.core.processing.ReadingModule`)
        * Writing Module (:class:`pynpoint.core.processing.WritingModule`)
        * Processing Module (:class:`pynpoint.core.processing.ProcessingModule`)

    Each PypelineModule has a name as a unique identifier in the Pypeline and requires the
    *connect_database* and *run* pipeline methods.
    """

    def __init__(self,
                 name_in):
        """
        Abstract constructor of a PypelineModule which needs a name as identifier.

        :param name_in: The name of the PypelineModule.
        :type name_in: str

        :return: None
        """

        assert (isinstance(name_in, str)), "Name of the PypelineModule needs to be a string."

        self._m_name = name_in
        self._m_data_base = None
        self._m_config_port = ConfigPort("config")

    @property
    def name(self):
        """
        Returns the name of the PypelineModule. This property makes sure that the internal module
        name can not be changed.

        :return: The name of the PypelineModule.
        :rtype: str
        """

        return self._m_name

    @abstractmethod
    def connect_database(self,
                         data_base_in):
        """
        Abstract interface for the function *connect_database* which is needed to connect the Ports
        of a PypelineModule with the DataStorage.

        :param data_base_in: The central database.
        :type data_base_in: DataStorage
        """

        # pass

    @abstractmethod
    def run(self):
        """
        Abstract interface for the run method of a PypelineModule which inheres the actual
        algorithm behind the module.
        """

        # pass


class WritingModule(six.with_metaclass(ABCMeta, PypelineModule)):
    """
    The abstract class WritingModule is an interface for processing steps in the pipeline which
    do not change the content of the internal DataStorage. They only have reading access to the
    central data base. WritingModules can be used to export data from the HDF5 database.
    WritingModules know the directory on the hard drive where the output of the module can be
    saved. If no output directory is given the default Pypeline output directory is used.
    WritingModules have a dictionary of input ports (self._m_input_ports) but no output ports.
    """

    def __init__(self,
                 name_in,
                 output_dir=None):
        """
        Abstract constructor of a WritingModule which needs the unique name identifier as input
        (more information: :class:`pynpoint.core.processing.PypelineModule`). In addition one can
        specify a output directory where the module will save its results. If no output directory is
        given the Pypeline default directory is used. This function is called in all *__init__()*
        functions inheriting from this class.

        :param name_in: The name of the WritingModule.
        :type name_in: str
        :param output_dir: Directory where the results will be saved.
        :type output_dir: str

        :return: None
        """

        super(WritingModule, self).__init__(name_in)

        assert (os.path.isdir(str(output_dir)) or output_dir is None), 'Output directory for ' \
            'writing module does not exist - input requested: %s.' % output_dir

        self.m_output_location = output_dir
        self._m_input_ports = {}

    def add_input_port(self,
                       tag):
        """
        Function which creates an InputPort for a WritingModule and appends it to the internal
        InputPort dictionary. This function should be used by classes inheriting from WritingModule
        to make sure that only input ports with unique tags are added. The new port can be used
        as: ::

             port = self._m_input_ports[tag]

        or by using the returned Port.

        :param tag: Tag of the new input port.
        :type tag: str

        :return: The new InputPort for the WritingModule.
        :rtype: InputPort
        """

        port = InputPort(tag)

        if self._m_data_base is not None:
            port.set_database_connection(self._m_data_base)

        self._m_input_ports[tag] = port

        return port

    def connect_database(self,
                         data_base_in):
        """
        Function used by a WritingModule to connect all ports in the internal input and output
        port dictionaries to the database. The function is called by Pypeline and connects the
        DataStorage object to all module ports.

        :param data_base_in: The central database.
        :type data_base_in: DataStorage

        :return: None
        """

        for port in six.itervalues(self._m_input_ports):
            port.set_database_connection(data_base_in)

        self._m_config_port.set_database_connection(data_base_in)

        self._m_data_base = data_base_in

    def get_all_input_tags(self):
        """
        Returns a list of all input tags to the WritingModule.

        :return: list of input tags
        :rtype: list
        """

        return list(self._m_input_ports.keys())

    @abstractmethod
    def run(self):
        """
        Abstract interface for the run method of a WritingModule which inheres the actual
        algorithm behind the module.
        """

        # pass


class ProcessingModule(six.with_metaclass(ABCMeta, PypelineModule)):
    """
    The abstract class ProcessingModule is an interface for all processing steps in the pipeline
    which read, process, and store data. Hence processing modules have read and write access to the
    central database through a dictionary of output ports (self._m_output_ports) and a dictionary
    of input ports (self._m_input_ports).
    """

    def __init__(self,
                 name_in):
        """
        Abstract constructor of a ProcessingModule which needs the unique name identifier as input
        (more information: :class:`pynpoint.core.processing.PypelineModule`). Call this function in
        all __init__() functions inheriting from this class.

        :param name_in: The name of the Processing Module
        :type name_in: str
        """

        super(ProcessingModule, self).__init__(name_in)

        self._m_input_ports = {}
        self._m_output_ports = {}

    def add_input_port(self,
                       tag):
        """
        Function which creates an InputPort for a ProcessingModule and appends it to the internal
        InputPort dictionary. This function should be used by classes inheriting from
        ProcessingModule to make sure that only input ports with unique tags are added. The new
        port can be used as: ::

             port = self._m_input_ports[tag]

        or by using the returned Port.

        :param tag: Tag of the new input port.
        :type tag: str

        :return: The new InputPort for the ProcessingModule.
        :rtype: InputPort
        """

        port = InputPort(tag)

        if self._m_data_base is not None:
            port.set_database_connection(self._m_data_base)

        self._m_input_ports[tag] = port

        return port

    def add_output_port(self,
                        tag,
                        activation=True):
        """
        Function which creates an OutputPort for a ProcessingModule and appends it to the internal
        OutputPort dictionary. This function should be used by classes inheriting from
        ProcessingModule to make sure that only output ports with unique tags are added. The new
        port can be used as: ::

             port = self._m_output_ports[tag]

        or by using the returned Port.

        :param tag: Tag of the new output port.
        :type tag: str
        :param activation: Activation status of the Port after creation. Deactivated ports
                           will not save their results until they are activated.
        :type activation: bool

        :return: The new OutputPort for the ProcessingModule.
        :rtype: OutputPort
        """

        port = OutputPort(tag, activate_init=activation)

        if tag in self._m_output_ports:
            warnings.warn("Tag '%s' of ProcessingModule '%s' is already used."
                          % (tag, self._m_name))

        if self._m_data_base is not None:
            port.set_database_connection(self._m_data_base)

        self._m_output_ports[tag] = port

        return port

    def connect_database(self,
                         data_base_in):
        """
        Function used by a ProcessingModule to connect all ports in the internal input and output
        port dictionaries to the database. The function is called by Pypeline and connects the
        DataStorage object to all module ports.

        :param data_base_in: The central database.
        :type data_base_in: DataStorage

        :return: None
        """

        for port in six.itervalues(self._m_input_ports):
            port.set_database_connection(data_base_in)

        for port in six.itervalues(self._m_output_ports):
            port.set_database_connection(data_base_in)

        self._m_config_port.set_database_connection(data_base_in)

        self._m_data_base = data_base_in

    def apply_function_in_time(self,
                               func,
                               image_in_port,
                               image_out_port,
                               func_args=None):
        """
        Applies a function to all pixel lines in time.

        :param func: The input function.
        :type func: function
        :param image_in_port: InputPort which is linked to the input data.
        :type image_in_port: InputPort
        :param image_out_port: OutputPort which is linked to the result place.
        :type image_out_port: OutputPort
        :param func_args: Additional arguments which are needed by the function *func*.
        :type func_args: tuple

        :return: None
        """

        init_line = image_in_port[:, 0, 0]

        size = apply_function(init_line, func, func_args).shape[0]

        # if image_out_port.tag == image_in_port.tag and size != image_in_port.get_shape()[0]:
        #     raise ValueError("Input and output port have the same tag while %s is changing " \
        #         "the length of the signal. Use different input and output ports instead." % func)

        image_out_port.set_all(np.zeros((size,
                                         image_in_port.get_shape()[1],
                                         image_in_port.get_shape()[2])),
                               data_dim=3,
                               keep_attributes=False)

        cpu = self._m_config_port.get_attribute("CPU")

        line_processor = LineProcessingCapsule(image_in_port,
                                               image_out_port,
                                               cpu,
                                               func,
                                               func_args,
                                               size)

        line_processor.run()

    def apply_function_to_images(self,
                                 func,
                                 image_in_port,
                                 image_out_port,
                                 message,
                                 func_args=None):
        """
        Function which applies a function to all images of an input port. The MEMORY attribute
        from the central configuration is used to load subsets of images into the memory. Note
        that the function *func* is not allowed to change the shape of the images if the input
        and output port have the same tag and MEMORY is not None.

        :param func: The function which is applied to all images. Its definitions should be
                     similar to: ::

                         def function(image_in,
                                      parameter1,
                                      parameter2,
                                      parameter3)

        :type func: function
        :param image_in_port: InputPort which is linked to the input data.
        :type image_in_port: InputPort
        :param image_out_port: OutputPort which is linked to the result place. No data is written
                               if set to None.
        :type image_out_port: OutputPort
        :param message: Progress message that is printed.
        :type message: str
        :param func_args: Additional arguments which are needed by the function *func*.
        :type func_args: tuple

        :return: None
        """

        memory = self._m_config_port.get_attribute("MEMORY")

        def _initialize():
            """
            Internal function to get the number of dimensions and subdivide the images by the
            MEMORY attribute.

            :return: Number of dimensions and array with subdivision of the images.
            :rtype: int, numpy.ndarray
            """

            ndim = image_in_port.get_ndim()

            if ndim == 2:
                nimages = 1
            elif ndim == 3:
                nimages = image_in_port.get_shape()[0]

            if image_out_port is not None and image_out_port.tag != image_in_port.tag:
                image_out_port.del_all_attributes()
                image_out_port.del_all_data()

            frames = memory_frames(memory, nimages)

            return ndim, frames

        def _append_result(ndim, images):
            """
            Internal function to apply the function on the images and append the results to a list.


            :param ndim: Number of dimensions.
            :type ndim: int
            :param images: Stack of images.
            :type images: numpy.ndarray

            :return: List with results of the function.
            :rtype: list
            """

            result = []

            if func_args is None:
                if ndim == 2:
                    result.append(func(images))

                elif ndim == 3:
                    for k in six.moves.range(images.shape[0]):
                        result.append(func(images[k]))

            else:
                if ndim == 2:
                    result.append(func(images, * func_args))

                elif ndim == 3:
                    for k in six.moves.range(images.shape[0]):
                        result.append(func(images[k], * func_args))

            return np.asarray(result)

        ndim, frames = _initialize()

        for i, _ in enumerate(frames[:-1]):
            progress(i, len(frames[:-1]), message)

            if ndim == 2:
                images = image_in_port[:, :]
            elif ndim == 3:
                images = image_in_port[frames[i]:frames[i+1], ]

            result = _append_result(ndim, images)

            if image_out_port is not None:
                if image_out_port.tag == image_in_port.tag:
                    if image_in_port.get_shape()[-1] == result.shape[-1] and \
                        image_in_port.get_shape()[-2] == result.shape[-2]:

                        if np.size(frames) == 2:
                            image_out_port.set_all(result, keep_attributes=True)

                        else:
                            image_out_port[frames[i]:frames[i+1]] = result

                    else:
                        raise ValueError("Input and output port have the same tag while the input "
                                         "function is changing the image shape. This is only "
                                         "possible with MEMORY=None.")

                else:
                    if ndim == 2:
                        result = np.squeeze(result, axis=0)

                    image_out_port.append(result)

        sys.stdout.write(message+" [DONE]\n")
        sys.stdout.flush()

    def get_all_input_tags(self):
        """
        Returns a list of all input tags to the ProcessingModule.

        :return: List of input tags.
        :rtype: list(str)
        """

        return list(self._m_input_ports.keys())

    def get_all_output_tags(self):
        """
        Returns a list of all output tags to the ProcessingModule.

        :return: List of output tags.
        :rtype: list(str)
        """

        return list(self._m_output_ports.keys())

    @abstractmethod
    def run(self):
        """
        Abstract interface for the run method of a ProcessingModule which inheres the actual
        algorithm behind the module.
        """

        # pass


class ReadingModule(six.with_metaclass(ABCMeta, PypelineModule)):
    """
    The abstract class ReadingModule is an interface for processing steps in the Pypeline which
    have only read access to the central data storage. One can specify a directory on the hard
    drive where the input data for the module is located. If no input directory is given then
    default Pypeline input directory is used. Reading modules have a dictionary of output ports
    (self._m_out_ports) but no input ports.
    """

    def __init__(self,
                 name_in,
                 input_dir=None):
        """
        Abstract constructor of ReadingModule which needs the unique name identifier as input
        (more information: :class:`pynpoint.core.processing.PypelineModule`). An input directory
        can be specified for the location of the data or else the Pypeline default directory is
        used. This function is called in all *__init__()* functions inheriting from this class.

        :param name_in: The name of the ReadingModule.
        :type name_in: str
        :param input_dir: Directory where the input files are located.
        :type input_dir: str

        :return: None
        """

        super(ReadingModule, self).__init__(name_in)

        assert (os.path.isdir(str(input_dir)) or input_dir is None), 'Input directory for ' \
            'reading module does not exist - input requested: %s.' % input_dir

        self.m_input_location = input_dir
        self._m_output_ports = {}

    def add_output_port(self,
                        tag,
                        activation=True):
        """
        Function which creates an OutputPort for a ReadingModule and appends it to the internal
        OutputPort dictionary. This function should be used by classes inheriting from
        ReadingModule to make sure that only output ports with unique tags are added. The new
        port can be used as: ::

             port = self._m_output_ports[tag]

        or by using the returned Port.

        :param tag: Tag of the new output port.
        :type tag: str
        :param activation: Activation status of the Port after creation. Deactivated ports
                           will not save their results until they are activated.
        :type activation: bool

        :return: The new OutputPort for the ReadingModule.
        :rtype: OutputPort
        """

        port = OutputPort(tag, activate_init=activation)

        if tag in self._m_output_ports:
            warnings.warn("Tag '%s' of ReadingModule '%s' is already used."
                          % (tag, self._m_name))

        if self._m_data_base is not None:
            port.set_database_connection(self._m_data_base)

        self._m_output_ports[tag] = port

        return port

    def connect_database(self,
                         data_base_in):
        """
        Function used by a ReadingModule to connect all ports in the internal input and output
        port dictionaries to the database. The function is called by Pypeline and connects the
        DataStorage object to all module ports.

        :param data_base_in: The central database.
        :type data_base_in: DataStorage

        :return: None
        """

        for port in six.itervalues(self._m_output_ports):
            port.set_database_connection(data_base_in)

        self._m_config_port.set_database_connection(data_base_in)

        self._m_data_base = data_base_in

    def get_all_output_tags(self):
        """
        Returns a list of all output tags to the ReadingModule.

        :return: List of output tags.
        :rtype: list, str
        """

        return list(self._m_output_ports.keys())

    @abstractmethod
    def run(self):
        """
        Abstract interface for the run method of a ReadingModule which inheres the actual
        algorithm behind the module.
        """

        # pass
