import time
import cv2
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
from eagerx_interbotix.aruco_detector import ArucoPoseDetector


# Load camera intrinsic/extrinsics
CAM_PATH = "/home/r2ci/eagerx-dev/eagerx_interbotix/assets/calibrations"
CAM_INTRINSICS = "logitech_c170.yaml"
# CAM_EXTRINSICS = "eye_hand_calibration_2022-08-10-1757.yaml"
CAM_EXTRINSICS = "extrinsic.yaml"
with open(f"{CAM_PATH}/{CAM_INTRINSICS}", "r") as f:
    ci = yaml.safe_load(f)
with open(f"{CAM_PATH}/{CAM_EXTRINSICS}", "r") as f:
    ce = yaml.safe_load(f)
mtx = np.array(ci["camera_matrix"]["data"], dtype="float32").reshape(ci["camera_matrix"]["rows"], ci["camera_matrix"]["cols"])
dist = np.array(ci["distortion_coefficients"]["data"], dtype="float32").reshape(
    ci["distortion_coefficients"]["rows"], ci["distortion_coefficients"]["cols"]
)
height, width = ci["image_height"], ci["image_width"]
cam_translation = ce["camera_to_robot"]["translation"]
cam_rotation = ce["camera_to_robot"]["rotation"]

use_cam = True
aruco_id = 25
object_translation = np.array([[0], [0], [-0.05], [1.0]], dtype="float32")
marker_size = 0.08
font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text (below)

if use_cam:
    cam_index = 4  # 4
    cam = cv2.VideoCapture(cam_index)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Get image resolution
    ret = False
    while not ret:
        ret, frame = cam.read()
        if ret:
            height, width = frame.shape[:2]
else:
    IMG_PATH = "/home/r2ci/.config/JetBrains/PyCharm2021.3/scratches/aruco/test_images"

    import glob

    cv_img = []
    # for img in glob.glob(f"{IMG_PATH}/166014667*.jpg"):
    for img in glob.glob(f"{IMG_PATH}/*.jpg"):
        n = cv2.imread(img)
        height, width = n.shape[:2]
        cv_img.append(n)

    class DummyCam:
        def __init__(self, images, delay: float = 0.033):
            self.images = images
            self.counter = 0
            self.delay = delay

        def release(self):
            pass

        def read(self):
            img = self.images[self.counter]
            self.counter += 1
            self.counter = self.counter % len(self.images)
            time.sleep(self.delay)
            return True, img

    cam = DummyCam(cv_img, delay=1.0)


# Initialize detector
detector = ArucoPoseDetector(height, width, marker_size, mtx.tolist(), dist.tolist(), aruco_type="DICT_ARUCO_ORIGINAL")

while True:
    ret, image_raw = cam.read()

    # Continue of no image returned
    if not ret:
        continue

    # Undistort image
    image = detector.undistort(image_raw)

    # Get pose
    image, corners, ids, rvec, tvec = detector.estimate_pose(image, draw=False)

    if rvec is not None:
        mask = (ids == aruco_id)[:, 0]
        rvec = rvec[mask]
        tvec = tvec[mask]
        corners = tuple([c for c, m in zip(corners, mask) if m])
        if not mask.any():
            continue
        # todo: ArucoPoseDetector node
        T_c2b = np.zeros((4, 4), dtype="float32")
        T_c2b[3, 3] = 1
        T_c2b[:3, :3] = R.from_quat(cam_rotation).as_matrix()
        T_c2b[:3, 3] = cam_translation
        T_a2c = np.zeros((rvec.shape[0], 4, 4), dtype="float32")
        T_a2c[:, 3, 3] = 1
        # Set all rotation matrices
        rmat = R.from_rotvec(rvec[:, 0, :])
        T_a2c[:, :3, :3] = rmat.as_matrix()
        # Set all translations
        T_a2c[:, :3, 3] = tvec[:, 0, :]
        # Offset measurements
        pos_aic = (T_a2c @ object_translation[:, :])[:, :3, :]
        # Position aruco (with offset) in base frame (aib).
        pos_aib = T_c2b[:3, :3] @ pos_aic + T_c2b[:3, 3, None]
        # Rotation matrix from aruco to base frame
        rmat_a2c = R.from_rotvec(rvec[:, 0, :]).as_matrix()
        rmat_a2b = T_c2b[:3, :3] @ rmat_a2c
        quat_a2b = R.from_matrix(rmat_a2b).as_quat().astype("float32")
        # todo: WrappedYawSensor node
        orn = quat_a2b
        rmat_a2b = R.from_quat(orn).as_matrix()
        # Remove z axis from rotation matrix
        axis_idx = 2  # Upward pointing axis in base frame
        z_idx = np.argmax(np.abs(rmat_a2b[:, axis_idx, :]), axis=1)  # Take absolute value, if axis points downward.
        rmat_a2b_red = np.empty((rmat_a2b.shape[0], 2, 2), dtype="float32")
        # Calculate yaw rotation in base frame
        for i, idx in enumerate(z_idx):
            s = np.sign(rmat_a2b[i, axis_idx, idx])
            rmat_a2b_red[i, :, :] = np.delete(np.delete(rmat_a2b[i, :, :], obj=axis_idx, axis=0), obj=idx, axis=1)
            if (s > 0 and idx == 1) or (s < 0 and idx != 1):
                rmat_a2b_red[i, :, :] = rmat_a2b_red[i, :, :] @ np.array([[0, 1], [1, 0]], dtype="float32")[None, :, :]
        cos_yaw = rmat_a2b_red[:, 0, 0]
        sin_yaw = rmat_a2b_red[:, 1, 0]
        yaw_a2b = np.arctan2(sin_yaw, cos_yaw)
        # Wrap yaw to [0, pi/2]
        yaww_a2b = yaw_a2b % (np.pi / 2)
        # Calculate wrapped orientations around z axis in base frame
        cos_yaww = np.cos(yaww_a2b)
        sin_yaww = np.sin(yaww_a2b)
        rmat_a2b_w = np.zeros((len(yaww_a2b), 3, 3), dtype="float32")
        rmat_a2b_w[:, 0, 0] = cos_yaww
        rmat_a2b_w[:, 1, 1] = cos_yaww
        rmat_a2b_w[:, 1, 0] = sin_yaww
        rmat_a2b_w[:, 0, 1] = -sin_yaww
        rmat_a2b_w[:, 2, 2] = 1.0
        # todo: Back to camera frame (for plotting purposes)
        rmat_a2c_w = T_c2b[:3, :3].transpose()[None, :, :] @ rmat_a2b_w
        last_orientation = R.from_matrix(rmat_a2c_w).as_rotvec()[:, None, :]
        assert last_orientation.shape == rvec.shape, "Wrapped pose does not have the same shape as original rvec."
        z_offset = np.array([0, -0.01, 0])[None, None, :]
        detector.draw_pose(image, corners, rvec=last_orientation, tvec=tvec - z_offset, axis_length=0.02)
        for c, yw, pos in zip(corners, yaww_a2b, pos_aib[:, :, 0]):
            x, y = int(c[0, 0, 0]), int(c[0, 0, 1])
            # todo: Print positions in top left corner with enters in-between entries.
            cv2.putText(image, f"yaw={yw: .2f} | p={str(pos.round(2))}", (x, y), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            # cv2.putText(image, f"yaw={yw: .2f}", (x, y), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Display result frame
    cv2.imshow("image", image)

    key = cv2.waitKey(1)

    if key == 27:  # Press esc to exit
        print("esc break...")
        cam.release()
        cv2.destroyAllWindows()
        break

    if key == ord(" ") and use_cam:  # Press the spacebar to save
        filename = str(time.time())[:10] + ".jpg"
        cv2.imwrite(f"./test_images/{filename}", image_raw)
