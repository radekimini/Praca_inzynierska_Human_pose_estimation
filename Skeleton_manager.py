from multiprocessing import Process, Queue
from shared import setup_logger
from Skeleton import Skeleton
from Fake_data_sender import Fake_data_sender
from Visualizer_worker import Visualizer_worker

logger = setup_logger(__name__)

class Skeleton_manager:
    def __init__(self, path="calculated_points/projected_points.json", video_path="calculated_points/video.mp4"):
        self.queue_joints = Queue()
        self.queue_angles = Queue()
        self.queue_visual = Queue()

        self.faker = Fake_data_sender(path, video_path)
        self.skeleton = Skeleton()
        self.visualizer = Visualizer_worker()

    def start(self):
        try:
            logger.info("SkeletonManager starting processes")

            p_faker = Process(target=self.faker.fake_data_symulator, args=(self.queue_joints, self.queue_visual))
            p_skeleton = Process(target=self.skeleton.run, args=(self.queue_joints, self.queue_angles))
            p_visual = Process(target=self.visualizer.run, args=(self.queue_visual, self.queue_angles))

            p_faker.start()
            p_skeleton.start()
            p_visual.start()

            p_faker.join()
            p_skeleton.join()
            p_visual.join()
        except Exception as e:
            logger.error(f"SkeletonManager error: {e}")
