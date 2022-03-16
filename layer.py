class Layer():

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def get_output(self):
        if not len(self.output):
            raise Exception("No output")
        return self.output