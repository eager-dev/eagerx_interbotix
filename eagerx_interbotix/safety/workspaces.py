import pybullet as pyb
import pybullet_data


def cube_and_ground(client_id):
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)

    # ground plane
    ground_id = pyb.loadURDF(
        "plane.urdf",
        [0, 0, 0],
        useFixedBase=True,
        physicsClientId=client_id,
    )

    # some cubes for obstacles
    cube1_id = pyb.loadURDF("cube.urdf", [-0.35, 0, 0.15], useFixedBase=True, physicsClientId=client_id, globalScaling=0.25)

    # store body indices in a dict with more convenient key names
    bodies = {
        "ground": ground_id,
        "cube1": cube1_id,
    }
    return bodies


def exclude_behind_left_workspace(client_id):
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)

    # ground plane
    ground_id = pyb.loadURDF(
        "plane.urdf",
        [0, 0, 0],
        useFixedBase=True,
        physicsClientId=client_id,
    )

    # some cubes for obstacles
    cube1_id = pyb.loadURDF("cube.urdf", [-0.75, 0, 0], useFixedBase=True, physicsClientId=client_id, globalScaling=1.0)

    # some cubes for obstacles
    cube2_id = pyb.loadURDF("cube.urdf", [0, 0.75, 0], useFixedBase=True, physicsClientId=client_id, globalScaling=1.0)

    # store body indices in a dict with more convenient key names
    bodies = {
        "ground": ground_id,
        "cube1": cube1_id,
        "cube2": cube2_id,
    }
    return bodies


def exclude_ground(client_id):
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)

    # ground plane
    ground_id = pyb.loadURDF(
        "plane.urdf",
        [0, 0, 0],
        useFixedBase=True,
        physicsClientId=client_id,
    )

    # store body indices in a dict with more convenient key names
    bodies = {
        "ground": ground_id,
    }
    return bodies


def exclude_ground_minus_2cm(client_id):
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)

    # ground plane
    ground_id = pyb.loadURDF(
        "plane.urdf",
        [0, 0, -0.02],
        useFixedBase=True,
        physicsClientId=client_id,
    )

    # store body indices in a dict with more convenient key names
    bodies = {
        "ground": ground_id,
    }
    return bodies


def exclude_ground_minus_25mm(client_id):
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)

    # ground plane
    ground_id = pyb.loadURDF(
        "plane.urdf",
        [0, 0, -0.025],
        useFixedBase=True,
        physicsClientId=client_id,
    )

    # store body indices in a dict with more convenient key names
    bodies = {
        "ground": ground_id,
    }
    return bodies
