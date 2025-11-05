import time
from multiprocessing import Process, Queue
from shared import setup_logger
from Fake_data_sender import Fake_data_sender
from Skeleton import Skeleton

logger = setup_logger(__name__)

def run_process_skeleton(queue_joints,queue_angles):
    skel= Skeleton()
    try:
        logger.info("Skeleton start")
        skel.run(queue_joints,queue_angles)
    except Exception as e:
        logger.error(f"Error running Skeleton {e}")
    except KeyboardInterrupt:
        logger.info("End of Skeleton")


def run_process_faker(queue):
    faker = Fake_data_sender()
    try:
        logger.info("Data Simulator start")
        faker.fake_data_symulator(queue)
    except Exception as e:
        logger.error(f"Error running Data Simulator: {e}")
    except KeyboardInterrupt:
        logger.info("Data Simulator stop")

if __name__ == '__main__':
    logger.info("faker checker process start")
    time.sleep(1)


    try:
        queue_joints = Queue()
        queue_angles = Queue()
        processes = [Process(target=run_process_faker, args=(queue_joints,)), Process(target=run_process_skeleton, args=(queue_joints,queue_angles,))]

        for p in processes:
            p.start()
        for p in processes:
            p.join()
    except Exception as e:
        logger.error(f"Create main subprocesses: {e}")
