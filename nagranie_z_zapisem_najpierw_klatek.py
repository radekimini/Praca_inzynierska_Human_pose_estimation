import cv2
import time
import csv
import os
import subprocess
from pathlib import Path

def format_time_str(secs):
    ms = int((secs - int(secs)) * 1000)
    return time.strftime('%H:%M:%S', time.gmtime(secs)) + f'.{ms:03}'

def ffmpeg_available():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def nagrywaj(folder="nazwa folderu", csv_path="timestamps.csv", output_video="output.mp4"):
    kamera_index = 0
    cap = cv2.VideoCapture(kamera_index)

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Nie udało się otworzyć kamery.")
        return

    os.makedirs(folder, exist_ok=True)
    frame_number = 0
    recording = False
    timestamps = []
    start_time = 0

    print("SPACJĘ aby rozpocząć/zatrzymać nagrywanie, Q aby wyjść.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Błąd odczytu klatki.")
            break

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # SPACJA - toggle nagrywanie
            if not recording:
                print("Start nagrywania")
                start_time = time.time()
                timestamps = []
                frame_number = 0
                for f in Path(folder).glob("*.png"):
                    f.unlink()  # usuń stare pliki klatek
            else:
                print("Stop nagrywania")
                break
            recording = not recording

        elif key == ord('q'):
            print("Przerwano bez zapisu")
            cap.release()
            cv2.destroyAllWindows()
            return

        if recording:
            now = time.time()
            elapsed = now - start_time
            timestamp_str = format_time_str(elapsed)

            # Zapis klatki jako obraz
            filename = os.path.join(folder, f"{frame_number:06d}.png")

            # Zapis timestamp
            timestamps.append((filename, elapsed))

            # Zegar
            cv2.putText(frame, timestamp_str, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imwrite(filename, frame)

            frame_number += 1
        cv2.imshow("Nagrywanie", frame)

    cap.release()
    cv2.destroyAllWindows()

    # Zapisz CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["plik", "czas_od_startu_s"])
        for row in timestamps:
            writer.writerow([os.path.basename(row[0]), f"{row[1]:.6f}"])

    print(f"Zapisano {frame_number} klatek do folderu '{folder}'")
    print(f"Timestampy w: '{csv_path}'")

    list_file = os.path.join(folder, "frames.txt")
    with open(list_file, 'w') as f:
        for i in range(len(timestamps)):
            f.write(f"file '{os.path.basename(timestamps[i][0])}'\n")
            if i < len(timestamps) - 1:
                delta = timestamps[i+1][1] - timestamps[i][1]
                f.write(f"duration {delta:.6f}\n")

        f.write(f"file '{os.path.basename(timestamps[-1][0])}'\n")
    print(f"Zawartość frames.txt:")
    with open(list_file, 'r') as f:
        print(f.read())
    if not ffmpeg_available():
        print("FFmpeg nie jest dostępny.")
        return

    # Tworzenie finalnego filmu ze zmiennym FPS
    print("Tworzenie nagrania")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", "frames.txt",
        "-vsync", "vfr",
        "-pix_fmt", "yuv420p",
        output_video
    ], cwd=folder)

    print(f"Film zapisany: {os.path.abspath(os.path.join(folder, output_video))}")
if __name__ == "__main__":
    nagrywaj()
