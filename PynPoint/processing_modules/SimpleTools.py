from PynPoint.core import ProcessingModule


class CutAroundCenterModule(ProcessingModule):

    def __init__(self,
                 new_shape,
                 name_in="cut_around_center",
                 image_in_tag="im_arr",
                 image_out_tag="cut_im_arr",
                 number_of_images_in_memory=100):

        super(CutAroundCenterModule, self).__init__(name_in=name_in)

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_number_of_images_in_memory = number_of_images_in_memory
        self.m_shape = new_shape

    def run(self):

        def image_cutting(image_in,
                          shape_in):

            shape_of_input = image_in.shape

            if shape_in[0] > shape_of_input[0] or shape_in[1] > shape_of_input[1]:
                raise ValueError("Input frame resolution smaller than target image resolution.")

            x_off = (shape_of_input[0] - shape_in[0]) / 2
            y_off = (shape_of_input[1] - shape_in[1]) / 2
            return image_in[x_off: shape_in[0] + x_off, y_off:shape_in[1] + y_off]

        self.apply_function_to_images(image_cutting,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      func_args=(self.m_shape,),
                                      num_images_in_memory=self.m_number_of_images_in_memory)

        self.m_image_out_port.add_history_information("Cutting Resolution to",
                                                      str(self.m_shape))

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()


class CutAroundPositionModule(ProcessingModule):

    def __init__(self,
                 new_shape,
                 center_of_cut,
                 name_in="cut_around_center",
                 image_in_tag="im_arr",
                 image_out_tag="cut_im_arr",
                 number_of_images_in_memory=100):

        super(CutAroundPositionModule, self).__init__(name_in=name_in)

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_number_of_images_in_memory = number_of_images_in_memory
        self.m_shape = new_shape
        self.m_center_of_cut = center_of_cut

    def run(self):

        def image_cutting(image_in,
                          shape_in,
                          center_of_cut_in):

            shape_of_input = image_in.shape

            if shape_in[0] > shape_of_input[0] or shape_in[1] > shape_of_input[1]:
                raise ValueError("Input frame resolution smaller than target image resolution.")

            x_off = center_of_cut_in[0] - (shape_in[0] / 2)
            y_off = center_of_cut_in[1] - (shape_in[1] / 2)

            return image_in[x_off: shape_in[0] + x_off, y_off:shape_in[1] + y_off]

        self.apply_function_to_images(image_cutting,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      func_args=(self.m_shape, self.m_center_of_cut),
                                      num_images_in_memory=self.m_number_of_images_in_memory)

        self.m_image_out_port.add_history_information("Cutting Resolution to",
                                                      str(self.m_shape))

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_image_out_port.close_port()