import eagerx_interbotix

# Other
import h5py
import pathlib
from matplotlib import pyplot as plt

NAME = "HER_force_torque_2022-10-13-1836"
STEPS = 1_600_000
MODEL_NAME = f"rl_model_{STEPS}_steps"
ROOT_DIR = pathlib.Path(eagerx_interbotix.__file__).parent.parent.resolve()
LOG_DIR = ROOT_DIR / "logs" / f"{NAME}"
GRAPH_FILE = f"graph.yaml"

if __name__ == "__main__":
    # This example shows how to access data from a generated dataset

    dataset_size = 1000

    # Read dataset file
    f = h5py.File(LOG_DIR / f"dataset_{dataset_size}.hdf5", "r")

    # print keys
    print(f.keys())

    # load sample
    sample_id = 0
    # Get image
    img = f["img"][sample_id]
    # Get xyz coordinate of box
    pos = f["box_pos"][sample_id]
    # Get angle of box
    angle = f["box_yaw"][sample_id]
    # Get xyz coordinate of goal
    goal_pos = f["goal_pos"][sample_id]
    # Get angle of goal
    goal_angle = f["goal_yaw"][sample_id]

    # Visualize image
    plt.figure()
    plt.title(f"Box: {pos}, {angle}\nGoal: {goal_pos}, {goal_angle}")
    plt.imshow(img)
    plt.show()

    f.close()
