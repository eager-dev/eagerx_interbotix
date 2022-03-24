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
    def spec(spec: ConverterSpec, index=0, offset=0):
        # Initialize spec with default arguments
        spec.initialize(MirrorAction)
        params = dict(offset=offset, index=index)
        spec.config.update(params)

    def initialize(self, offset, index):
        self.offset = offset
        self.index = index

    def convert(self, msg):
        action = msg.data[self.index]
        mirrored = np.array([action, -action], dtype="float32")
        mirrored += self.offset
        return Float32MultiArray(data=mirrored)
