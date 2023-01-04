import eagerx
import eagerx_interbotix

# Other
import gym
import yaml
import gym.wrappers as w
import stable_baselines3 as sb
import os
import h5py
import pathlib
import torch
import torch.nn as nn
import numpy as np
import random
from torchvision.transforms import transforms
from PIL import Image


NAME = "HER_force_torque_2022-10-13-1836"
STEPS = 1_600_000
MODEL_NAME = f"rl_model_{STEPS}_steps"
ROOT_DIR = pathlib.Path(eagerx_interbotix.__file__).parent.parent.resolve()
LOG_DIR = ROOT_DIR / "logs" / f"{NAME}"
GRAPH_FILE = f"graph.yaml"


def predict_privileged_observation(
    model: torch.nn.Module,
    env: gym.Env,
    obs: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Predict the privileged observation from the given observation.

    Args:
        model (torch.nn.Module): The model to use for prediction.
        env (gym.Env): The environment.
        obs (np.ndarray): The observation to predict the privileged observation from.

    Returns:
        np.ndarray: The predicted privileged observation.
    """
    valid_render = False
    while not valid_render:
        rgb = env.render("rgb_array")
        valid_render = len(rgb.shape) > 0 and rgb.shape[0] > 0
    img = Image.fromarray(np.asarray(rgb, dtype="uint8"))
    t_img = input_transforms(img)
    y = model(t_img.unsqueeze(0).to(device))
    prediction = (y.cpu().numpy()[0] * target_std) + target_mean
    box_xy = prediction[:2]
    box_z = 0.05
    box_yaw = prediction[2]
    goal_xy = prediction[3:5]
    goal_z = 0.0
    goal_yaw = prediction[5]
    obs["achieved_goal"] = np.array([box_xy[0], box_xy[1], box_z, box_yaw])
    obs["desired_goal"] = np.array([goal_xy[0], goal_xy[1], goal_z, goal_yaw])
    return obs


if __name__ == "__main__":
    eagerx.set_log_level(eagerx.WARN)

    episodes = 100
    # Set seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    date_time = "2023-01-04_15-14"
    dataset_size = 20000
    batch_size = 32
    epochs = 10
    model_seed = 1
    model_split = 0.7
    model_path = str(LOG_DIR / f"{date_time}" / f"{dataset_size}_{batch_size}_{epochs}_{model_split}_{model_seed}.tar")

    # Load model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pytorch_model = torch.hub.load("pytorch/vision:v0.10.0", "shufflenet_v2_x1_0", pretrained=False)
    # Adjust last layer
    pytorch_model.fc = nn.Linear(1024, 6)
    for param in pytorch_model.parameters():
        param.requires_grad = False
    pytorch_model.to(device)

    checkpoint = torch.load(model_path, map_location=device)  # map from gpu to cpu
    epoch = checkpoint['epoch']
    pytorch_model.load_state_dict(checkpoint['model_state_dict'])
    train_loss = checkpoint['train_loss']
    test_loss = checkpoint['test_loss']
    image_mean = checkpoint['image_mean']
    image_std = checkpoint['image_std']
    target_mean = checkpoint['target_mean']
    target_std = checkpoint['target_std']
    pytorch_model.eval()

    # Create input transforms
    input_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=image_mean,
                std=image_std,
            ),
        ]
    )


    CAM_PATH = ROOT_DIR / "assets" / "calibrations"
    CAM_INTRINSICS = "logitech_c920.yaml"
    with open(f"{CAM_PATH}/{CAM_INTRINSICS}", "r") as f:
        cam_intrinsics = yaml.safe_load(f)

    sync = True
    must_render = True
    T_max = 15.0  # [sec]
    rate = 10
    render_rate = rate
    safe_rate = 10
    dataset_size = 10000
    image_width, image_height = 64, 64
    light_direction_low = [-50, -50, 0]
    light_direction_high = [50, 50, 50]
    robot_color_low = [0, 0, 0, 1]
    robot_color_high = [0.2, 0.2, 0.2, 1]
    box_color_low = [0.9*1, 0.9*0.388, 0.9*0.278, 1]
    box_color_high = [1, 1.1*0.388, 1.1*0.278, 1]
    goal_color_low = [0.9*0.278, 0.9*1, 0.9*0.388, 1]
    goal_color_high = [1.1*0.278, 1, 1.1*0.388, 1]

    # Camera settings
    cam_index_ov = 1
    cam_translation_ov = [0.8, 0, 0.8]  # todo: set correct cam overview location
    cam_rotation_ov = [-0.6830127, -0.6830127, 0.1830127, 0.1830127]  # todo: set correct cam overview location
    cam_render_shape = [image_height, image_width]

    # Create dataset file
    f = h5py.File(LOG_DIR / f"dataset_{dataset_size}.hdf5", "w")
    image_dataset = f.create_dataset("img", (dataset_size, image_height, image_width, 3), dtype="uint")
    boxpos_dataset = f.create_dataset("box_pos", (dataset_size, 3), dtype="float")
    boxyaw_dataset = f.create_dataset("box_yaw", (dataset_size, 1), dtype="float")
    goalpos_dataset = f.create_dataset("goal_pos", (dataset_size, 3), dtype="float")
    goalyaw_dataset = f.create_dataset("goal_yaw", (dataset_size, 1), dtype="float")

    # Load graph
    graph = eagerx.Graph.load(f"{LOG_DIR}/{GRAPH_FILE}")

    # Use correct workspace in safety filter
    safe = graph.get_spec("safety")
    safe.config.collision.workspace = "eagerx_interbotix.safety.workspaces/exclude_ground"

    # Set color robot
    vx300s = graph.get_spec("vx300s")
    vx300s.states.color.space.update(low=robot_color_low, high=robot_color_high)

    # Modify aruco rate
    solid = graph.get_spec("solid")
    solid.sensors.robot_view.rate = 10
    solid.sensors.position.rate = 10
    solid.sensors.orientation.rate = 10
    solid.config.cam_intrinsics = cam_intrinsics
    solid.states.color.space.update(low=box_color_low, high=box_color_high)


    # Modify goal
    goal = graph.get_spec("goal")
    x, y, z = 0.30, 0.0, 0.05
    dx, dy = 0.1, 0.20
    solid.states.orientation.space.update(low=[-1, -1, 0, 0], high=[1, 1, 0, 0])
    solid.states.position.space.update(low=[x, -y - dy, z], high=[x + dx, y + dy, z])
    goal.states.orientation.space.update(low=[-1, -1, 0, 0], high=[1, 1, 0, 0])
    goal.states.position.space.update(low=[x, -y - dy, 0], high=[x + dx, y + dy, 0])
    goal.states.color.space.update(low=goal_color_low, high=goal_color_high)

    # Add rendering
    if must_render:
        # Create camera
        from eagerx_interbotix.camera.objects import Camera

        cam = Camera.make(
            "cam",
            rate=render_rate,
            sensors=["image"],
            urdf=os.path.dirname(eagerx_interbotix.__file__) + "/camera/assets/realsense2_d435.urdf",
            optical_link="camera_color_optical_frame",
            calibration_link="camera_color_optical_frame",
            camera_index=cam_index_ov,
            render_shape=cam_render_shape,
            fov=45.0,
            light_direction_low=light_direction_low,
            light_direction_high=light_direction_high,
        )
        graph.add(cam)
        cam.states.pos.space.update(low=cam_translation_ov, high=cam_translation_ov)
        cam.states.orientation.space.update(low=cam_rotation_ov, high=cam_rotation_ov)


        # Connect
        graph.render(source=cam.sensors.image, rate=render_rate, encoding="bgr")

    # Define engines
    from eagerx_pybullet.engine import PybulletEngine

    engine = PybulletEngine.make(rate=safe_rate, gui=True, egl=True, sync=True, real_time_factor=0.0)

    # Add surface
    surfuce_urdf = ROOT_DIR / "eagerx_interbotix" / "solid" / "assets" / "surface.urdf"
    engine.add_object("surface", urdf=str(surfuce_urdf), baseOrientation=[0, 0, 0, 1])

    # Make backend
    from eagerx.backends.single_process import SingleProcess

    SingleProcess.MIN_THREADS = 100
    backend = SingleProcess.make()

    # Define environment
    from eagerx_interbotix.env import ArmEnv
    from eagerx_interbotix.goal_env import GoalArmEnv

    # Initialize env
    env = ArmEnv(
        name="ArmEnv",
        rate=rate,
        graph=graph,
        engine=engine,
        backend=backend,
        add_bias=False,
        exclude_z=False,
        max_steps=int(T_max * rate),
    )
    sb_env = GoalArmEnv(env, add_bias=False)
    sb_env = w.rescale_action.RescaleAction(sb_env, min_action=-1.0, max_action=1.0)

    # Load model
    model = sb.SAC.load(f"{LOG_DIR}/{MODEL_NAME}", sb_env, verbose=1)

    # Evaluate
    for i in range(episodes):
        step = 0
        print(f"Episode {i+1}/{episodes}")
        obs, done, frames = sb_env.reset(), False, []
        obs = predict_privileged_observation(obs=obs, env=env, model=pytorch_model, target_mean=target_mean, target_std=target_std, device=device)
        episodic_rewards = []
        episodic_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = sb_env.step(action)
            obs = predict_privileged_observation(obs=obs, env=env, model=pytorch_model, target_mean=target_mean, target_std=target_std, device=device)
            step += 1
            episodic_reward += reward
        episodic_rewards.append(episodic_reward)
        print(f"Episode reward: {episodic_reward}")
    print(f"Average reward: {np.mean(episodic_rewards)}")
    f.close()
