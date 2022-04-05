import os
import eagerx_interbotix
import rospy
import rosparam
import yaml
from eagerx.utils.node_utils import launch_node
from eagerx.utils.utils import get_param_with_blocking


def get_configs(robot_model: str, motor_config: str, mode_config: str):
    module_path = os.path.dirname(eagerx_interbotix.__file__) + "/../assets/"
    config_path = module_path + "config/"
    try:
        mode_config = mode_config if isinstance(mode_config, str) else f"{config_path}/modes.yaml"
        with open(mode_config, "r") as yamlfile:
            mode_config = yaml.safe_load(yamlfile)
    except IOError:
        rospy.logerr(f"Mode Config File was not found in: {config_path}")
        raise

    try:
        motor_config = motor_config if isinstance(motor_config, str) else f"{config_path}/{robot_model}.yaml"
        with open(motor_config, "r") as yamlfile:
            motor_config = yaml.safe_load(yamlfile)
    except IOError:
        rospy.logerr(f"Motor Config File was not found in: {config_path}")
        raise
    return motor_config, mode_config


def generate_urdf(
    robot_model: str,
    ns="",
    base_link_frame="base_link",
    show_ar_tag=False,
    show_gripper_bar=True,
    show_gripper_fingers=True,
    use_world_frame=True,
    external_urdf_loc="",
    load_gazebo_configs=False,
):
    module_path = os.path.dirname(eagerx_interbotix.__file__) + "/../assets/"
    launch_file = module_path + "xsarm_description.launch"
    cli_args = [
        f"module_path:={module_path}",
        f"robot_model:={robot_model}",
        f"ns:={ns}",
        f"base_link_frame:={base_link_frame}",
        f"show_ar_tag:={show_ar_tag}",
        f"show_gripper_bar:={show_gripper_bar}",
        f"show_gripper_fingers:={show_gripper_fingers}",
        f"use_world_frame:={use_world_frame}",
        f"external_urdf_loc:={external_urdf_loc}",
        f"load_gazebo_configs:={load_gazebo_configs}",
    ]
    launch = launch_node(launch_file, cli_args)
    launch.start()

    # Replace mesh urls
    urdf_key = f"{ns}/{robot_model}/robot_description"
    urdf: str = get_param_with_blocking(urdf_key)
    urdf_sbtd = urdf.replace("package://interbotix_xsarm_descriptions/", module_path)
    rosparam.upload_params(urdf_key, urdf_sbtd)

    return urdf_key

    # Write to tmp file
    #
