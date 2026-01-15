import time
import subprocess
from multiprocessing import Process, Queue
from shared import setup_logger
from Skeleton import Skeleton
# from Fake_data_sender import Fake_data_sender
from Visualizer_worker import Visualizer_worker
from demo.demo import  main as d_main



logger = setup_logger(__name__)

if __name__ == '__main__':
    logger.info("FakeDataSenderRun: starting simulation with video and projected points")
    time.sleep(1)

    # pogram_robot= subprocess.Popen("odpal_najpierw_ruszanie_robotem.exe")
    # time.sleep(2)

    path="calculated_points/projected_points.json"
    video_path="calculated_points/video.mp4"
    queue_joints = Queue()
    queue_angles = Queue()
    queue_visual = Queue()

    # faker = Fake_data_sender(path, video_path)
    # siec = d_main()
    skeleton = Skeleton()
    visualizer = Visualizer_worker()


    try:
        logger.info("SkeletonManager starting processes")

        p_siec = Process(target=d_main, args=(queue_joints, queue_visual))
        p_skeleton = Process(target=skeleton.run, args=(queue_joints, queue_angles))
        p_visual = Process(target=visualizer.run, args=(queue_visual, queue_angles))

        p_siec.start()
        p_skeleton.start()
        p_visual.start()

        p_siec.join()
        p_skeleton.join()
        p_visual.join()
    except Exception as e:
        logger.error(f"SkeletonManager error: {e}")
#
# import socket
# robot_ip = '192.168.0.100'
# port = 49938
# local_port = 49939
# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sock.bind(('',local_port))
# sock.sendto(b'\x00'*10,(robot_ip,port))
# print("czekam na odpowiedz")
# data,addr = sock.recvfrom(4096)
# print(f"odpowiedz z {addr}: {data}")