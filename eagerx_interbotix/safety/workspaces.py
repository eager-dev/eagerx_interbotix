import pybullet as pyb
import pybullet_data


def cubes_and_2dof(client_id):
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)

    # ground plane
    ground_id = pyb.loadURDF(
        "plane.urdf",
        [0, 0, 0],
        useFixedBase=True,
        physicsClientId=client_id,
    )

    # some cubes for obstacles
    cube1_id = pyb.loadURDF(
        "cube.urdf",
        [0.5, 0, 0.15],
        useFixedBase=True,
        physicsClientId=client_id,
        globalScaling=0.25
    )

    # store body indices in a dict with more convenient key names
    bodies = {
        "ground": ground_id,
        "cube1": cube1_id,
    }
    return bodies