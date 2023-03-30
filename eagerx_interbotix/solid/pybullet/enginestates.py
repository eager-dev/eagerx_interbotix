import eagerx
from eagerx.core.specs import EngineStateSpec
from typing import Dict


class TextureState(eagerx.EngineState):
    @classmethod
    def make(cls, texture_path) -> EngineStateSpec:
        """A spec to create an EngineState that resets a specified link to the desired color.
        :return: EngineStateSpec
        """
        spec = cls.get_specification()
        spec.config.update(texture_path=texture_path)
        return spec

    def initialize(self, spec: EngineStateSpec, simulator: Dict):
        """Initializes the engine state according to the spec."""
        self.robot = simulator["object"]
        p = simulator["client"]
        x = p.loadTexture(spec.config.texture_path)
        for _pb_name, part in self.robot.parts.items():
            bodyid, linkindex = part.get_bodyid_linkindex()
            p.changeVisualShape(self.robot.robot_objectid[0], linkIndex=-1, textureUniqueId=x)

    def reset(self, state):
        """Resets the link state to the desired value."""
        pass
