import time
import numpy as np
from shared import read_config, setup_logger

logger = setup_logger(__name__)

# processes = [Process(target=run_process_adc, args=(queue,)), Process(target=run_process_pcc_listener),
#              Process(target=run_process_telemetry, args=(queue,))]
class Fake_data_sender:
    def __init__(self):
        try:
            self.target_duration = 1
            self.points = read_config("calculated_points/projected_points.json")
            fixed_data = []
            for frame in self.points:
                frame_fixed = frame.copy()
                while len(frame_fixed) < 16:
                    frame_fixed.append([np.nan, np.nan])  # lub [None, None]
                fixed_data.append(frame_fixed)

            self.skeleton_calculated = np.array(fixed_data)
            del fixed_data
            del frame_fixed
        except Exception as e:
            logger.error(f"Fake data sender cannot init: {e}")

    def fake_data_symulator(self, queue):
        for index, frame in enumerate(self.skeleton_calculated):
            try:
                start = time.perf_counter()

                queue.put(frame)
                print(f"frame {index}:\n {frame}")

                elapsed = time.perf_counter() - start
                remaining = self.target_duration - elapsed
                if remaining > 0:
                    time.sleep(remaining)
            except Exception as e:
                logger.error(f"Error queueing data in faker in frame {index}: {e}")
