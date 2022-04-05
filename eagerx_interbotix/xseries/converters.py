# ROS IMPORTS
from std_msgs.msg import Float32MultiArray

# RX IMPORTS
from eagerx.core import register as register
from eagerx.core.entities import Processor
from eagerx.core.specs import ConverterSpec

# OTHER
import numpy as np


class MirrorAction(Processor):
    MSG_TYPE = Float32MultiArray

    @staticmethod
    @register.spec("MirrorAction", Processor)
    def spec(spec: ConverterSpec, index=0, constant=0, offset=0, scale=1):
        # Initialize spec with default arguments
        spec.initialize(MirrorAction)
        params = dict(offset=offset, index=index, scale=scale, constant=constant)
        spec.config.update(params)

    def initialize(self, offset, index, scale, constant):
        self.offset = offset
        self.index = index
        self.scale = scale
        self.constant = constant

    def convert(self, msg):
        action = self.scale * msg.data[self.index] + self.constant
        mirrored = np.array([action, -action], dtype="float32")
        mirrored += self.offset
        return Float32MultiArray(data=mirrored)
