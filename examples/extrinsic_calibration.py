import eagerx
import yaml
import numpy as np
from collections import deque


CAM_PATH = "/home/r2ci/eagerx-dev/eagerx_interbotix/assets/calibrations"
CAM_EXTRINSICS = "eye_hand_calibration_2022-08-11-1659_gripper_link.yaml"


if __name__ == "__main__":
    eagerx.set_log_level(eagerx.WARN)

    # Load camera intrinsic/extrinsics
    with open(f"{CAM_PATH}/{CAM_EXTRINSICS}", "r") as f:
        ce = yaml.safe_load(f)
    marker_rotation = ce["marker_to_ee"]["rotation"]
    marker_translation = ce["marker_to_ee"]["translation"]

    rate = 10

    # Initialize empty graph
    graph = eagerx.Graph.create()

    # Add arm
    from eagerx_interbotix.xseries.xseries import Xseries
    robot_type = "vx300s"
    arm = Xseries.make(
        name=robot_type,
        robot_type=robot_type,
        sensors=["position", "velocity", "ee_pos", "ee_orn"],
        actuators=["pos_control"],
        states=["position", "velocity", "gripper"],
        rate=rate,
    )
    arm.states.gripper.space.update(low=[0.], high=[0.])  # Set gripper to closed position
    arm.actuators.pos_control.space.low = [0. for _ in arm.actuators.pos_control.space.low]
    arm.actuators.pos_control.space.high = [0. for _ in arm.actuators.pos_control.space.high]

    # Point wrist downward:
    arm.actuators.pos_control.space.low[4] = 1.0
    arm.actuators.pos_control.space.high[4] = 1.0
    graph.add(arm)

    # Add camera
    CAM_INTRINSICS = "logitech_camera.yaml"
    cam_index = 2

    from eagerx_interbotix.camera.objects import Camera
    with open(f"{CAM_PATH}/{CAM_INTRINSICS}", "r") as f:
        ci = yaml.safe_load(f)
    render_shape = [ci["image_height"], ci["image_width"]]
    cam = Camera.make("cam",
                      rate=rate,
                      render_shape=render_shape,
                      camera_index=cam_index)
    graph.add(cam)

    # Add aruco detector
    from eagerx_interbotix.camera.calibration import CameraCalibrator
    cali = CameraCalibrator.make("calibration",
                                 rate=rate,
                                 aruco_id=26,
                                 aruco_size=0.04,
                                 aruco_type="DICT_ARUCO_ORIGINAL",
                                 marker_translation=marker_translation,
                                 marker_rotation=marker_rotation,
                                 cam_intrinsics=ci)
    graph.add(cali)

    # Connect graph
    graph.connect(source=arm.sensors.ee_pos, target=cali.inputs.ee_position)
    graph.connect(source=arm.sensors.ee_orn, target=cali.inputs.ee_orientation)
    graph.connect(source=cam.sensors.image, target=cali.inputs.image)
    graph.connect(source=cali.outputs.translation, observation="cam_translation")
    graph.connect(source=cali.outputs.orientation, observation="cam_orientation")
    graph.connect(action="position", target=arm.actuators.pos_control)
    graph.render(source=cali.outputs.image_aruco, rate=rate)

    # Define environment
    class CalibrationEnv(eagerx.BaseEnv):
        def __init__(self, name, rate, graph, engine, backend, force_start, max_steps: int):
            self.steps = 0
            self.max_steps = max_steps
            self.translation = deque(maxlen=int(rate * 10))
            self.orientation = deque(maxlen=int(rate * 10))
            super().__init__(name, rate, graph, engine, backend=backend, force_start=force_start)

        def step(self, action):
            # Step the environment
            self.steps += 1
            info = dict()
            obs = self._step(action)

            # Get cam translation & rotation
            cam_translation = obs["cam_translation"][0]
            cam_orientation = obs["cam_orientation"][0]

            # Print solid location
            if not np.abs(cam_translation).sum() < 0.01:
                self.translation.append(cam_translation)
                self.orientation.append(cam_orientation)

                # Create np arrays of window
                trans_np = np.array(self.translation)
                orn_np = np.array(self.orientation)

                # Apply median filter
                median_trans = np.median(trans_np, axis=0)
                approx_median_orn = np.median(orn_np, axis=0)
                norm_median_orn = approx_median_orn / np.linalg.norm(approx_median_orn)
                median_orn = orn_np[np.argmin(np.linalg.norm(orn_np - norm_median_orn, axis=1))]

                msg = f"translation={str(median_trans.round(3))} | rotation={str(median_orn.round(3))}"
                print(msg)
            return obs, 0., False, info

        def reset(self):
            # Reset steps counter
            self.steps = 0

            # Sample states
            states = self.state_space.sample()

            # Perform reset
            obs = self._reset(states)
            return obs

    # Make engine
    from eagerx_reality.engine import RealEngine
    engine = RealEngine.make(rate=rate, sync=True)

    # Make backend
    # from eagerx.backends.ros1 import Ros1
    # backend = Ros1.make()
    from eagerx.backends.single_process import SingleProcess
    backend = SingleProcess.make()

    # Initialize env
    env = CalibrationEnv(name="CalibrationEnv", rate=rate, graph=graph, engine=engine, backend=backend, force_start=True,
                         max_steps=100000)
    env.render()

    # Evaluate
    for eps in range(5000):
        print(f"Episode {eps}")
        obs, done = env.reset(), False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
