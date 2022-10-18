import eagerx
import numpy as np
from eagerx.core.specs import ProcessorSpec


class MirrorAction(eagerx.Processor):
    @classmethod
    def make(cls, index=0, constant=0, offset=0, scale=1) -> ProcessorSpec:
        spec = cls.get_specification()
        spec.config.update(offset=offset, index=index, scale=scale, constant=constant)
        return spec

    def initialize(self, spec: ProcessorSpec):
        self.offset = spec.config.offset
        self.index = spec.config.index
        self.scale = spec.config.scale
        self.constant = spec.config.constant

    def convert(self, msg):
        action = self.scale * msg.data[self.index] + self.constant
        mirrored = np.array([action, -action], dtype="float32")
        mirrored += self.offset
        return mirrored
