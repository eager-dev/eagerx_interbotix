import os
import yaml
from eagerx.utils.utils_sub import substitute_args


class XmlListConfig(list):
    """copied from: https://stackoverflow.com/questions/2148119/how-to-convert-an-xml-string-to-a-dictionary"""

    def __init__(self, aList):
        for element in aList:
            if element:
                # treat like dict
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                # treat like list
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)


class XmlDictConfig(dict):
    """
    copied from: https://stackoverflow.com/questions/2148119/how-to-convert-an-xml-string-to-a-dictionary

    Example usage:

    # >>> tree = ElementTree.parse('your_file.xml')
    # >>> root = tree.getroot()
    # >>> xmldict = XmlDictConfig(root)

    Or, if you want to use an XML string:

    # >>> root = ElementTree.XML(xml_string)
    # >>> xmldict = XmlDictConfig(root)

    And then use xmldict for what it is... a dict.
    """

    def __init__(self, parent_element):
        if parent_element.items():
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                # treat like dict - we assume that if the first two tags
                # in a series are different, then they are all different.
                if len(element) == 1 or element[0].tag != element[1].tag:
                    aDict = XmlDictConfig(element)
                # treat like list - we assume that if the first two tags
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is the
                    # tag name the list elements all share in common, and
                    # the value is the list itself
                    aDict = {element[0].tag: XmlListConfig(element)}
                # if the tag has attributes, add those to the dict
                if element.items():
                    aDict.update(dict(element.items()))
                self.update({element.tag: aDict})
            # this assumes that if you've got an attribute in a tag,
            # you won't be having any text. This may or may not be a
            # good idea -- time will tell. It works for the way we are
            # currently doing XML configuration files...
            elif element.items():
                self.update({element.tag: dict(element.items())})
            # finally, if there are no child tags and no attributes, extract
            # the text
            else:
                self.update({element.tag: element.text})


def launch_node(launch_file, args):
    import roslaunch

    cli_args = [substitute_args(launch_file)] + args
    roslaunch_args = cli_args[1:]
    roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    # roslaunch.configure_logging(uuid)  # THIS RESETS the log level. Can we do without this line? Are ROS logs stil being made?
    launch = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
    return launch


def get_configs(robot_model: str, motor_config: str, mode_config: str):
    import rospy

    module_path = os.path.dirname(__file__) + "/xseries/assets"
    assert os.path.exists(module_path), f"Module path does not exist: {module_path}"
    config_path = module_path + "/config"
    assert os.path.exists(config_path), f"Config path does not exist: {config_path}"
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
    import rosparam
    import rospy

    module_path = os.path.dirname(__file__) + "/xseries/assets/"
    assert os.path.exists(module_path), f"Module path does not exist: {module_path}"
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
    urdf: str = rospy.get_param(urdf_key)
    urdf_sbtd = urdf.replace("package://interbotix_xsarm_descriptions/", module_path)
    rosparam.upload_params(urdf_key, urdf_sbtd)

    return urdf_key
