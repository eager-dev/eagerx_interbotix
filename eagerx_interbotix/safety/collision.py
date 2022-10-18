# Adapted from github:@AdamHeins (02-04-2022)
# https://adamheins.com/blog/collision-detection-pybullet
# https://github.com/adamheins/pyb_utils/blob/4c610925acea69976cb5232169004af4c4de541a/pyb_utils/collision.py

from dataclasses import dataclass

import numpy as np
import pybullet as pyb


@dataclass
class NamedCollisionObject:
    """Name of a body and one of its links.
    The body name must correspond to the key in the `bodies` dict, but is
    otherwise arbitrary. The link name should match the URDF. The link name may
    also be None, in which case the base link (index -1) is used.
    """

    body_name: str
    link_name: str = None


@dataclass
class IndexedCollisionObject:
    """Index of a body and one of its links."""

    body_uid: int
    link_uid: int


@dataclass
class IndexedJointObject:
    """Index of a robot joint and its name."""

    joint_name: str
    joint_uid: int


def index_collision_pairs(physics_uid, bodies, named_collision_pairs):
    """Convert a list of named collision pairs to indexed collision pairs.
    In other words, convert named bodies and links to the indexes used by
    PyBullet to facilitate computing collisions between the objects.
    Parameters:
      physics_uid: Index of the PyBullet physics server to use.
      bodies: dict with body name keys and corresponding indices as values
      named_collision_pairs: a list of 2-tuples of NamedCollisionObject
    Returns: a list of 2-tuples of IndexedCollisionObject
    """

    # build a nested dictionary mapping body names to link names to link
    # indices
    body_link_map = {}
    for name, uid in bodies.items():
        body_link_map[name] = {}
        n = pyb.getNumJoints(uid, physics_uid)
        for i in range(n):
            info = pyb.getJointInfo(uid, i, physics_uid)
            link_name = info[12].decode("utf-8")
            body_link_map[name][link_name] = i

    def _index_named_collision_object(obj):
        """Map body and link names to corresponding indices."""
        body_uid = bodies[obj.body_name]
        if obj.link_name is not None:
            link_uid = body_link_map[obj.body_name][obj.link_name]
        else:  # todo: then make collision objects for each link.
            link_uid = -1
        return IndexedCollisionObject(body_uid, link_uid)

    # convert all pairs of named collision objects to indices
    indexed_collision_pairs = []
    for a, b in named_collision_pairs:
        a_indexed = _index_named_collision_object(a)
        b_indexed = _index_named_collision_object(b)
        indexed_collision_pairs.append((a_indexed, b_indexed))

    return indexed_collision_pairs


def index_joints(physics_uid, robot_id, joints):
    """Map a list of joint names to indexed joints.
    In other words, map named joints to the index used by
    PyBullet to facilitate setting the configuration.
    Parameters:
      physics_uid: Index of the PyBullet physics server to use.
      joints: list with joint name keys
    Returns: a list of IndexedJointObject
    """
    indexed_joints = []
    n = pyb.getNumJoints(robot_id, physics_uid)
    for joint_name in joints:
        for i in range(n):
            info = pyb.getJointInfo(robot_id, i, physics_uid)
            if joint_name == info[1].decode("utf-8"):
                indexed_joints.append(IndexedJointObject(joint_name, i))
                continue
    assert len(joints) == len(indexed_joints), "Not all joints were found in the provided urdf."
    return indexed_joints


class CollisionDetector:
    def __init__(self, col_id, bodies, joints, named_collision_pairs):
        self.ds = None
        self.col_id = col_id
        self.robot_id = bodies["robot"]
        self.indexed_collision_pairs = index_collision_pairs(self.col_id, bodies, named_collision_pairs)
        self.indexed_joints = index_joints(self.col_id, self.robot_id, joints)

    def compute_distances(self, q=None, max_distance=1.0):
        """Compute closest distances for a given configuration.
        Parameters:
          q: Iterable representing the desired configuration. This is applied
             directly to PyBullet body with index bodies["robot"].
          max_distance: Bodies farther apart than this distance are not queried
             by PyBullet, the return value for the distance between such bodies
             will be max_distance.
        Returns: A NumPy array of distances, one per pair of collision objects.
        """

        # put the robot in the given configuration
        if q is not None:
            for i, joint in enumerate(self.indexed_joints):
                pyb.resetJointState(self.robot_id, joint.joint_uid, q[i], physicsClientId=self.col_id)

        # compute shortest distances between all object pairs
        distances = []
        for a, b in self.indexed_collision_pairs:
            closest_points = pyb.getClosestPoints(
                a.body_uid,
                b.body_uid,
                distance=max_distance,
                linkIndexA=a.link_uid,
                linkIndexB=b.link_uid,
                physicsClientId=self.col_id,
            )

            # if bodies are above max_distance apart, nothing is returned, so
            # we just saturate at max_distance. Otherwise, take the minimum
            if len(closest_points) == 0:
                distances.append(max_distance)
            else:
                distances.append(np.min([pt[8] for pt in closest_points]))

        return np.array(distances)

    def in_collision(self, q=None, margin=0):
        """Returns True if configuration q is in collision, False otherwise.
        Parameters:
          q: Iterable representing the desired configuration.
          margin: Distance at which objects are considered in collision.
             Default is 0.0.
        """
        self.ds = self.compute_distances(q=q, max_distance=margin * 2)
        return (self.ds < margin).any()

    def get_distance(self):
        return self.ds


def get_robot_link_clusters(urdf, joints):
    clusters = {
        j: dict(links=[], connects=[]) for j in joints
    }  # {nf_joint: links=["l_1", "l_2"], connects=["joint_1", "join_2"]}
    world_cluster = "base"
    assert world_cluster not in joints, f"Cannot have a joint name '{world_cluster}'. It is a reserved joint name."
    clusters[world_cluster] = dict(links=[], connects=[])
    pm = urdf.parent_map
    cm = urdf.child_map

    def find_nonfixed_parent_joint(link):
        parent_joint, parent_link = pm[link]
        if parent_link not in pm:
            return world_cluster
        if parent_joint in joints:
            return parent_joint
        else:
            return find_nonfixed_parent_joint(parent_link)

    for link, (parent_joint, _parent_link) in urdf.parent_map.items():
        nf_parent_joint = find_nonfixed_parent_joint(link)
        if urdf.link_map[link].collision is not None:
            clusters[nf_parent_joint]["links"].append(link)
        clusters[nf_parent_joint]["connects"].append(parent_joint)
        if link in cm:
            for child_joint, _ in cm[link]:
                clusters[nf_parent_joint]["connects"].append(child_joint)

    for _, cluster in clusters.items():
        cluster["connects"] = list(set(cluster["connects"]))
    return clusters


def get_self_collision_pairs(clusters):
    key_pairs = {}
    for j_1, c_1 in clusters.items():
        key_pairs[j_1] = []
        for j_2, c_2 in clusters.items():
            if j_2 not in c_1["connects"] and j_1 not in c_2["connects"]:
                if j_2 not in key_pairs:
                    key_pairs[j_1].append(j_2)
    pairs = []
    for j_1, js in key_pairs.items():
        for l_1 in clusters[j_1]["links"]:
            nco_1 = NamedCollisionObject("robot", l_1)
            for j_2 in js:
                for l_2 in clusters[j_2]["links"]:
                    nco_2 = NamedCollisionObject("robot", l_2)
                    pairs.append((nco_1, nco_2))

    return pairs


def get_workspace_collision_pairs(bodies, clusters):
    pairs = []
    named_bodies = [NamedCollisionObject(b) for b in bodies if b != "robot"]
    for joint, cluster in clusters.items():
        for link in cluster["links"]:
            nco_1 = NamedCollisionObject("robot", link)
            for nco_2 in named_bodies:
                if joint == "base" and nco_2.body_name == "ground":
                    continue
                pairs.append((nco_1, nco_2))
    return pairs


def get_named_collision_pairs(bodies, urdf, joints):
    clusters = get_robot_link_clusters(urdf, joints)
    # Get self collision pairs
    self_collision_pairs = get_self_collision_pairs(clusters)
    # Get workspace pairs
    workspace_pairs = get_workspace_collision_pairs(bodies, clusters)
    return self_collision_pairs, workspace_pairs
