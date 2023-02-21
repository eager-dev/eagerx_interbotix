import typing as t
import cv2
import PIL
import numpy as np
import eagerx
from eagerx import Space
from eagerx.core.specs import NodeSpec
import eagerx.core.register as register
from eagerx.utils.utils import Msg
from scipy.spatial.transform import Rotation as R


class Overlay(eagerx.Node):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        cam_intrinsics: t.Dict,
        cam_extrinsics: t.Dict,
        process: int = eagerx.NEW_PROCESS,
        ratio: float = 0.2,
        resolution: t.List[int] = None,
        caption: str = "",
    ) -> NodeSpec:
        """
        Filters goal joint positions that cause self-collisions or are below a certain height.
        Also check velocity limits.

        :param name: Node name
        :param rate: Rate at which callback is called.
        :param process: {0: NEW_PROCESS, 1: ENVIRONMENT, 2: ENGINE, 3: EXTERNAL}
        :param ratio: The thumbnail:main resolution ratio.
        :param resolution: Resolution of the rendered image [height, width].
        :param caption: Text to place underneath thumbnail.
        :return: Node specification.
        """
        spec = cls.get_specification()

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.color = "grey"
        spec.config.inputs = ["main", "thumbnail", "goal_ori", "goal_pos"]
        spec.config.outputs = ["image"]
        spec.config.ratio = ratio
        spec.config.resolution = resolution if isinstance(resolution, list) else [480, 480]
        spec.config.caption = caption
        spec.config.cam_intrinsics = cam_intrinsics
        spec.config.cam_extrinsics = cam_extrinsics
        return spec

    def initialize(self, spec: NodeSpec):
        caption = spec.config.caption
        border_px = 10

        ci = spec.config.cam_intrinsics
        camera_matrix = np.array(ci["camera_matrix"]["data"], dtype="float32").reshape(
            ci["camera_matrix"]["rows"], ci["camera_matrix"]["cols"]
        )
        dist_coeffs = np.array(ci["distortion_coefficients"]["data"], dtype="float32").reshape(
            ci["distortion_coefficients"]["rows"], ci["distortion_coefficients"]["cols"]
        )
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        # Get homogeneous transformation matrix from camera to robot
        cam_translation = spec.config.cam_extrinsics["camera_to_robot"]["translation"]
        cam_rotation = spec.config.cam_extrinsics["camera_to_robot"]["rotation"]
        cam_rotation = R.from_quat(cam_rotation).as_matrix().transpose()
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[:3, :3] = cam_rotation
        homogeneous_matrix[:3, 3] = - cam_rotation @ cam_translation
        self.homogeneous_matrix = homogeneous_matrix
        self.ratio = spec.config.ratio
        self.h, self.w = spec.config.resolution
        self.h_tn, self.w_tn = int(self.h * self.ratio), int(self.w * self.ratio)
        h_tnb, w_tnb = self.h_tn + border_px * 2, self.w_tn + border_px * 2

        # Determine caption size
        font = cv2.FONT_HERSHEY_SIMPLEX
        color_cap = (255, 255, 255)
        thickness = 2  # It is the thickness of the line in px
        font_scale = 0.5  # Font scale factor that is multiplied by the font-specific base size.
        w_cap, h_cap = cv2.getTextSize(caption, font, fontScale=font_scale, thickness=thickness)[0]
        h_tnbc, w_tnbc = h_tnb + h_cap, w_tnb
        x_cap, y_cap = (w_tnbc - w_cap) // 2, h_tnbc - 4
        assert x_cap > 0, "Text too wide. Shorten text or choose a smaller font size."
        assert y_cap > 0, "Text too tall. Choose a smaller font size."
        tn = 0 * np.ones((h_tnbc, w_tnbc, 3), dtype="uint8")
        tn = cv2.putText(tn, caption, (x_cap, y_cap), font, fontScale=font_scale, thickness=thickness, color=color_cap)
        self.tn_bg = PIL.Image.fromarray(tn).convert("RGB")

        # Top-left corner (to fill tn_bg with thumbnail PIL image)
        self.x_tn_bg, self.y_tn_bg = border_px, border_px
        self.x_mn, self.y_mn = int(self.w - w_tnb), 0

    @register.states()
    def reset(self):
        pass

    @register.inputs(main=Space(dtype="uint8"), thumbnail=Space(dtype="uint8"), goal_pos=Space(dtype="float32"), goal_ori=Space(dtype="float32"))
    @register.outputs(image=Space(dtype="uint8"))
    def callback(self, t_n: float, main: Msg, thumbnail: Msg, goal_pos: Msg, goal_ori: Msg):
        mn = main.msgs[-1]
        tn = thumbnail.msgs[-1]
        goal_pos = goal_pos.msgs[-1]
        goal_pos[2] = 0.0
        goal_ori = goal_ori.msgs[-1]

        # Draw goal frame
        if goal_pos is not None and goal_ori is not None:
            # draw goal square
            p1 = goal_pos + 1.2 * np.array([-0.05, -0.05, 0.0])
            p2 = goal_pos + 1.2 * np.array([0.05, -0.05, 0.0])
            p3 = goal_pos + 1.2 * np.array([0.05, 0.05, 0.0])
            p4 = goal_pos + 1.2 * np.array([-0.05, 0.05, 0.0])
            p1 = self.homogeneous_matrix[:3, :3] @ p1 + self.homogeneous_matrix[:3, 3]
            p2 = self.homogeneous_matrix[:3, :3] @ p2 + self.homogeneous_matrix[:3, 3]
            p3 = self.homogeneous_matrix[:3, :3] @ p3 + self.homogeneous_matrix[:3, 3]
            p4 = self.homogeneous_matrix[:3, :3] @ p4 + self.homogeneous_matrix[:3, 3]

            p1 = cv2.projectPoints(p1.reshape(1, 3), np.zeros((3, 1)), np.zeros((3, 1)), self.camera_matrix, self.dist_coeffs)[0][0][0]
            p2 = cv2.projectPoints(p2.reshape(1, 3), np.zeros((3, 1)), np.zeros((3, 1)), self.camera_matrix, self.dist_coeffs)[0][0][0]
            p3 = cv2.projectPoints(p3.reshape(1, 3), np.zeros((3, 1)), np.zeros((3, 1)), self.camera_matrix, self.dist_coeffs)[0][0][0]
            p4 = cv2.projectPoints(p4.reshape(1, 3), np.zeros((3, 1)), np.zeros((3, 1)), self.camera_matrix, self.dist_coeffs)[0][0][0]
            cv2.line(mn, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)
            cv2.line(mn, (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1])), (0, 255, 0), 2)
            cv2.line(mn, (int(p3[0]), int(p3[1])), (int(p4[0]), int(p4[1])), (0, 255, 0), 2)
            cv2.line(mn, (int(p4[0]), int(p4[1])), (int(p1[0]), int(p1[1])), (0, 255, 0), 2)

        # Resize main
        mn_PIL = PIL.Image.fromarray(mn).convert("RGB")
        mn_PIL = mn_PIL.resize((self.w, self.h), PIL.Image.ANTIALIAS)

        # Resize thumbnail
        tn_PIL = PIL.Image.fromarray(tn).convert("RGB")
        tn_PIL = tn_PIL.resize((self.w_tn, self.h_tn), PIL.Image.ANTIALIAS)
        self.tn_bg.paste(tn_PIL, (self.x_tn_bg, self.y_tn_bg))

        # place thumbnail inside bordered image
        mn_PIL.paste(self.tn_bg, (self.x_mn, self.y_mn))
        image = np.array(mn_PIL).astype(tn.dtype)

        return dict(image=image)
