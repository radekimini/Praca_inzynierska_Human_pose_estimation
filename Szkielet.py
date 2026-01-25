import math
import csv
import socket
import struct
import time
from numpy import array, dot, linalg, clip
from shared import setup_logger, Joint

logger = setup_logger(__name__)


class Skeleton:

    def __init__(self, config_path="config.json"):
        # self.points = read_config(config_path)
        # self.points = np.array([np.array([0, 0]) for x in range(16)])
        # self.skel_vec = np.array([np.array(point) for point in self.points])
        self.skel_angles = {"L_ELBOW": 0.0, "R_ELBOW": 0.0, "L_ARM": 0.0, "R_ARM": 0.0}
        self.skel_vec = array([array([0, 0]) for x in range(16)])

        self.axis = 1
        self.delta = [0.0, 0.0]
        self.ruszaj = False

    def setup_io(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("127.0.0.1", 5000))
        self.csv_file = open("python_commands.csv", "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        # nagłówek
        self.csv_writer.writerow(["time", "event", "axis", "delta"])

    def send(self):
        data = struct.pack("if", self.axis, self.delta)
        self.sock.sendall(data)
        logger.info(f"[PY] SENT axis={self.axis}, delta={self.delta}, time={time.time()}")
        self.csv_writer.writerow([self.now(), "SEND", self.axis, self.delta])
        self.csv_file.flush()

    def wait_for_response(self):
        status = struct.unpack("i", self.sock.recv(4))[0]
        if status == 1:
            logger.info(f"[PY] Robot STARTED movement, time={time.time()}")
            self.csv_writer.writerow([self.now(), "STARTED", self.axis, self.delta])
        elif status == 2:
            logger.info(f"[PY] Robot FINISHED movement, time={time.time()}")
            self.csv_writer.writerow([self.now(), "FINISHED", self.axis, self.delta])
        elif status == -1:
            logger.warning(f"[PY] Robot REJECTED command, time={time.time()}")

    def move_robot(self):
        """
            ustala oś ruchu oraz prędkość poruszania robota, wysyła polecenie do procesu robota
        """
        try:
            self.ruszaj = False

            if self.skel_angles["LEGS"] <= 15.0:
                return

            legs_angle = min(self.skel_angles["LEGS"], 30.0)
            self.delta = (legs_angle / 750.0) - (1.0 / 50.0)

            #  GÓRA
            if (
                    self.skel_angles["L_ARM"] < 45.0 and
                    165.0 < self.skel_angles["L_ELBOW"] < 190.0 and
                    self.skel_angles["R_ARM"] > 165.0 and
                    165.0 < self.skel_angles["R_ELBOW"] < 190.0
            ):
                self.axis = 11
                self.ruszaj = True
                print("ruszanie gora")

            #  DÓŁ
            elif (
                    self.skel_angles["R_ARM"] < 45.0 and
                    165.0 < self.skel_angles["R_ELBOW"] < 190.0 and
                    self.skel_angles["L_ARM"] > 165.0 and
                    165.0 < self.skel_angles["L_ELBOW"] < 190.0
            ):
                self.axis = 11
                self.delta *= -1.0
                self.ruszaj = True
                print("ruszanie dol")

            #  TYŁ
            elif (
                    self.skel_angles["L_ARM"] < 45.0 and
                    165.0 < self.skel_angles["L_ELBOW"] < 190.0 and
                    80.0 < self.skel_angles["R_ARM"] < 110.0 and
                    165.0 < self.skel_angles["R_ELBOW"] < 190.0
            ):
                self.axis = 3
                self.ruszaj = True
                print("ruszanie tyl")

            #  PRZÓD
            elif (
                    self.skel_angles["R_ARM"] < 45.0 and
                    165.0 < self.skel_angles["R_ELBOW"] < 190.0 and
                    80.0 < self.skel_angles["L_ARM"] < 110.0 and
                    165.0 < self.skel_angles["L_ELBOW"] < 190.0
            ):
                self.axis = 3
                self.delta *= -1.0
                self.ruszaj = True
                print("ruszanie przod")

            #  LEWO 
            elif (
                    self.skel_angles["R_ARM"] < 45.0 and
                    80.0 < self.skel_angles["L_ARM"] < 110.0 and
                    80.0 < self.skel_angles["L_ELBOW"] < 110.0
            ):
                self.axis = 7
                self.delta *= -1.0
                self.ruszaj = True
                print("ruszanie lewo")

            #  PRAWO
            elif (
                    self.skel_angles["L_ARM"] < 45.0 and
                    80.0 < self.skel_angles["R_ARM"] < 110.0 and
                    80.0 < self.skel_angles["R_ELBOW"] < 110.0
            ):
                self.axis = 7
                self.ruszaj = True
                print("ruszanie prawo")

            if self.ruszaj:
                self.send()
                self.wait_for_response()
                self.wait_for_response()
                self.delta = 0.0

        except Exception as e:
            logger.error(f"move_robot error: {e}")

    def print_skeleton_points(self):
        """
        wyświetla koordynaty wszytskich punktów charakterystycznych
        """
        output = ""
        for index, vect in enumerate(self.skel_vec):
            output += f"{Joint(index).name} x_cord: {vect[0]}, y_cord: {vect[1]}\n"
        print(output)

    def print_skeleton_angles(self):
        """
        wyświetla wartosci wszytskich kątów
        """
        output = ""
        for key, angle in self.skel_angles.items():
            output += f"angle {key}: {angle}\n"
        print(output)

    def calculate_angle_between_segments(self, joint_a1: Joint, joint_a2: Joint, joint_b1: Joint, joint_b2: Joint):
        """
        Oblicza kąt w wierzchołku łączącym dwa segmenty.
        Dla wyprostowanych segmentów zwróci 180 stopni.
        """

        vec_a = self.skel_vec[joint_a1.value] - self.skel_vec[joint_a2.value]
        vec_b = self.skel_vec[joint_b2.value] - self.skel_vec[joint_b1.value]

        len_a = linalg.norm(vec_a)
        len_b = linalg.norm(vec_b)

        if len_a == 0 or len_b == 0:
            return None

        dot_product = dot(vec_a, vec_b)
        # Cosinus kąta między wektorami
        cos_theta = dot_product / (len_a * len_b)

        cos_theta = clip(cos_theta, -1.0, 1.0)

        angle_rad = math.acos(cos_theta)
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def update_skeleton(self, queue_frames, queue_angles):
        """
        główny proces
        oblicza kąty i wysyła koleja dalej, uruchamia proces obliczania sterowania
        """
        while True:
            try:
                packet = queue_frames.get()

                self.skel_vec = packet["points"]

                self.skel_angles["L_ELBOW"] = self.calculate_angle_between_segments(
                    Joint.L_HAND, Joint.L_ELBOW,
                    Joint.L_ELBOW, Joint.L_SHOULDER
                )
                self.skel_angles["R_ELBOW"] = self.calculate_angle_between_segments(
                    Joint.R_HAND, Joint.R_ELBOW,
                    Joint.R_ELBOW, Joint.R_SHOULDER
                )
                self.skel_angles["L_ARM"] = self.calculate_angle_between_segments(
                    Joint.L_HIP, Joint.L_SHOULDER,
                    Joint.L_SHOULDER, Joint.L_ELBOW
                )
                self.skel_angles["R_ARM"] = self.calculate_angle_between_segments(
                    Joint.R_HIP, Joint.R_SHOULDER,
                    Joint.R_SHOULDER, Joint.R_ELBOW
                )
                self.skel_angles["LEGS"] = self.calculate_angle_between_segments(
                    Joint.R_FOOT, Joint.R_HIP,
                    Joint.L_HIP, Joint.L_FOOT
                )

                while not queue_angles.empty():
                    queue_angles.get()
                queue_angles.put(self.skel_angles.copy())

                self.move_robot()

            except Exception as e:
                logger.error(f"Skeleton error: {e}")

    def run(self, queue_frames,queue_angles):
        logger.info("starting skeleton")
        try:
            self.setup_io()
            self.update_skeleton(queue_frames,queue_angles)
        except Exception as e:
            logger.error(f"Error starting update skeleton: {e}")
