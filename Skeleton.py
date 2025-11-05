import math
from numpy import array, dot, linalg, clip
from enum import Enum
from shared import setup_logger

logger = setup_logger(__name__)


class Joint(Enum):
    R_FOOT = 0
    R_KNEE = 1
    R_HIP = 2
    L_HIP = 3
    L_KNEE = 4
    L_FOOT = 5
    C_HIP = 6
    C_SHOULDER = 7
    NECK = 8
    HEAD = 9
    R_HAND = 10
    R_ELBOW = 11
    R_SHOULDER = 12
    L_SHOULDER = 13
    L_ELBOW = 14
    L_HAND = 15


class Skeleton:

    def __init__(self, config_path="config.json"):
        # self.points = read_config(config_path)
        # self.points = np.array([np.array([0, 0]) for x in range(16)])
        # self.skel_vec = np.array([np.array(point) for point in self.points])
        self.skel_angles = {"L_ELBOW": 0.0, "R_ELBOW": 0.0, "L_ARM": 0.0, "R_ARM": 0.0}
        self.skel_vec = array([array([0, 0]) for x in range(16)])

    def print_skeleton_points(self):
        """
        Prints cords of all nodes
        """
        output = ""
        for index, vect in enumerate(self.skel_vec):
            output += f"{Joint(index).name} x_cord: {vect[0]}, y_cord: {vect[1]}\n"
        print(output)

    def print_skeleton_angles(self):
        """
        Prints cords of all nodes
        """
        output = ""
        for key, angle in self.skel_angles.items():
            output += f"angle {key}: {angle}\n"
        print(output)

    def calculate_angle_between_segments(self, joint_a1: Joint, joint_a2: Joint, joint_b1: Joint, joint_b2: Joint):
        """
        Calculates the angle between two segments:
        segment A: joint_a1 → joint_a2
        segment B: joint_b1 → joint_b2
        Returns angle in degrees (0..180)
        """

        # Create vectors from the given joints
        vec_a = self.skel_vec[joint_a2.value] - self.skel_vec[joint_a1.value]
        vec_b = self.skel_vec[joint_b2.value] - self.skel_vec[joint_b1.value]

        # Calculate lengths
        len_a = linalg.norm(vec_a)
        len_b = linalg.norm(vec_b)

        if len_a == 0 or len_b == 0:
            return None

        # Calculate angle
        dot_product = dot(vec_a, vec_b)
        cos_theta = dot_product / (len_a * len_b)

        # Clamp to avoid numerical errors
        cos_theta = clip(cos_theta, -1.0, 1.0)

        angle_rad = math.acos(cos_theta)
        angle_deg = math.degrees(angle_rad)

        # if angle_deg > 90:
        #     angle_deg -= 90

        return angle_deg

    def update_skeleton(self, queue_joints, queue_angles):
        """
        updating skeleton_vectors inside class
        """

        while True:
            while not queue_joints.empty():
                try:
                    data = queue_joints.get()
                    self.skel_vec = array([array(point) for point in data])
                    del data
                    print("\nreceived queue data\n")
                    # self.print_skeleton_points()
                except Exception as e:
                    logger.error(f"Error receiving queue data: {e}")

                try:

                    self.skel_angles["L_ELBOW"] = self.calculate_angle_between_segments(Joint.L_HAND, Joint.L_ELBOW,
                                                                                        Joint.L_ELBOW, Joint.L_SHOULDER)
                    self.skel_angles["R_ELBOW"] = self.calculate_angle_between_segments(Joint.R_HAND, Joint.R_ELBOW,
                                                                                        Joint.R_ELBOW, Joint.R_SHOULDER)
                    self.skel_angles["L_ARM"] = self.calculate_angle_between_segments(Joint.C_HIP, Joint.C_SHOULDER,
                                                                                      Joint.L_SHOULDER, Joint.L_HAND)
                    self.skel_angles["R_ARM"] = self.calculate_angle_between_segments(Joint.C_HIP, Joint.C_SHOULDER,
                                                                                      Joint.R_SHOULDER, Joint.R_HAND)

                    # self.print_skeleton_angles()

                except Exception as e:
                    logger.error(f"Error in calculation of angles : {e}")
                try:
                    queue_angles.put(self.skel_angles)

                except Exception as e:
                    logger.error(f"Error in queuing data to robot : {e}")

    def run(self, queue_joints, queue_angles):
        logger.info("starting skeleton")
        try:
            self.update_skeleton(queue_joints, queue_angles)
        except Exception as e:
            logger.error(f"Error starting update skeleton: {e}")

        # try:
        #     update_process = Process(target=self.update_skeleton, args=(queue,))
        # except Exception as e:
        #     logger.error(f"Error starting Processes: {e}")
        # try:
        #     update_process.start()
        #
        #     update_process.join()
        # except Exception as e:
        #     logger.error(f"Error joining processes Processes: {e}")
