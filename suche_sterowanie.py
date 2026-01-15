import socket
import struct
import time
import csv
from datetime import datetime


SERVER_IP = "127.0.0.1"
SERVER_PORT = 5000
SEND_INTERVAL = 0.5


def now():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log(msg):
    print(f"[{now()}] {msg}")


def recv_status(sock):
    data = sock.recv(4)
    if len(data) != 4:
        raise RuntimeError("Connection closed by server")
    return struct.unpack("i", data)[0]


def main():
    log("Connecting to C++ server...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_IP, SERVER_PORT))
    log("Connected.\n")

    csv_file = open("python_commands.csv", "a", newline="")
    csv_writer = csv.writer(csv_file)

    # nagłówek
    csv_writer.writerow(["time", "event", "axis", "delta"])

    while True:
        try:
            print("\n--- NEW TEST ---")
            iterations = int(input("Podaj ilość iteracji (np. 5): "))
            axis = int(input("Podaj numer axisa (3, 7 lub 11): "))
            delta = float(input("Podaj deltę [m] (np. 0.02): "))

            if axis not in (3, 7, 11):
                log("❌ BŁĄD: dozwolone osie to tylko 3, 7, 11")
                continue

            log(f"START TESTU: iterations={iterations}, axis={axis}, delta={delta}")

            for i in range(iterations):
                csv_writer.writerow([now(), "SEND", axis, delta])
                csv_file.flush()

                log(f" Iteracja {i + 1}/{iterations} – wysyłam polecenie")

                cmd = struct.pack("if", axis, delta)
                sock.sendall(cmd)
                log(f" SENT axis={axis}, delta={delta}")

                status = recv_status(sock)

                if status == 1:
                    log("Robot STARTED ruch")
                    csv_writer.writerow([now(), "STARTED", axis, delta])
                elif status == -1:
                    log("Robot REJECTED polecenie (busy)")
                    time.sleep(SEND_INTERVAL)
                    continue
                else:
                    log(f"Nieznany status: {status}")
                csv_file.flush()
                # --- CZEKAJ NA ZAKOŃCZENIE ---
                status = recv_status(sock)
                if status == 2:
                    log("Robot FINISHED ruch")
                    csv_writer.writerow([now(), "FINISHED", axis, delta])
                else:
                    log(f"Nieoczekiwany status końcowy: {status}")
                csv_file.flush()

                time.sleep(SEND_INTERVAL)

            log("TEST ZAKOŃCZONY")

        except KeyboardInterrupt:
            log("Zamykanie programu (Ctrl+C)")
            break
        except Exception as e:
            log(f" BŁĄD: {e}")
            break

    sock.close()
    log("Rozłączono.")


if __name__ == "__main__":
    main()
