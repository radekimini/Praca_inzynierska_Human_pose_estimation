import time
from multiprocessing import Process, Queue
from shared import setup_logger
from Skeleton import Skeleton
from Fake_data_sender import Fake_data_sender
from Visualizer_worker import Visualizer_worker

logger = setup_logger(__name__)

if __name__ == '__main__':
    logger.info("FakeDataSenderRun: starting simulation with video and projected points")
    time.sleep(1)


    path="calculated_points/projected_points.json"
    video_path="calculated_points/video.mp4"
    queue_joints = Queue()
    queue_angles = Queue()
    queue_visual = Queue()

    faker = Fake_data_sender(path, video_path)
    skeleton = Skeleton()
    visualizer = Visualizer_worker()


    try:
        logger.info("SkeletonManager starting processes")

        p_faker = Process(target=faker.fake_data_symulator, args=(queue_joints, queue_visual))
        p_skeleton = Process(target=skeleton.run, args=(queue_joints, queue_angles))
        p_visual = Process(target=visualizer.run, args=(queue_visual, queue_angles))

        p_faker.start()
        p_skeleton.start()
        p_visual.start()

        p_faker.join()
        p_skeleton.join()
        p_visual.join()
    except Exception as e:
        logger.error(f"SkeletonManager error: {e}")
