
#define WIN32_LEAN_AND_MEAN
#include <iostream>
#include <fstream>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <chrono>
#include <iomanip>
#include <sstream>


#pragma comment(lib, "ws2_32.lib")

#include "include/FastResearchInterface.h"

#include <atomic>

std::atomic<bool> newCommandArrived(false);
static int counter = 0;
float MAX_MOVE_TIME = 0.2f; // [s] czas trwania ruchu
std::ofstream csv_cmd("cpp_commands.csv", std::ios::app);


std::string now()
{
    using namespace std::chrono;

    auto tp = system_clock::now();
    auto t = system_clock::to_time_t(tp);
    auto ms = duration_cast<milliseconds>(tp.time_since_epoch()) % 1000;

    std::tm bt{};
    localtime_s(&bt, &t);

    std::ostringstream oss;
    oss << std::put_time(&bt, "%H:%M:%S")
        << "." << std::setw(3) << std::setfill('0') << ms.count();

    return oss.str();
}


struct Command
{
    int axis;
    float delta;
};

struct Response {
    int status;   // 1 = STARTED, 2 = FINISHED, -1 = REJECTED
};

enum class RobotState
{
    IDLE,
    MOVING
};

RobotState robotState = RobotState::IDLE;


bool receiveCommand(SOCKET client, int& axis, float& delta)
{
    static Command cmd;   // zapamiętany bufor
    int received = recv(client, (char*)&cmd, sizeof(cmd), 0);

    if (received == sizeof(Command))
    {

        axis = cmd.axis;
        delta = cmd.delta;
        std::cout << "[C++] time=" << now()
            << " event=RECEIVE axis=" << axis << "\n";

        csv_cmd << now() << ",RECEIVE," << axis << "," << delta << "\n";
        csv_cmd.flush();

        return true;   // nowe polecenie
        
    }

    if (received == 0)
        return false;  // rozłączenie

    // received < 0 to brak danych
    return false;
}

void setInitialStiffness(FastResearchInterface& fri)
{
    float stiffness[6] = { 2500.0f, 2500.0f, 2500.0f, 150.0f, 150.0f, 150.0f };
    float damping[6] = { 0.7f,    0.7f,    0.7f,    0.7f,    0.7f,    0.7f };

    fri.SetCommandedCartStiffness(stiffness);
    fri.SetCommandedCartDamping(damping);
}


bool moveAxis(FastResearchInterface& fri, int axisIndex, float delta, SOCKET client)
{
    robotState = RobotState::MOVING;

    // WYŚLIJ "STARTED" 
    int started = 1;
    send(client, (char*)&started, sizeof(int), 0);

    float startPose[12];
    float commandPose[12];
    fri.GetMeasuredCartPose(startPose);

    for (int i = 0; i < 12; ++i)
        commandPose[i] = startPose[i];

    const float cycleTime = fri.GetFRICycleTime();
    const int steps = static_cast<int>(MAX_MOVE_TIME / cycleTime);
    csv_cmd << now() << ",ACCEPTED," << axisIndex << "," << delta << "\n";
    csv_cmd.flush();


    for (int i = 0; i < steps; ++i)
    {
        fri.WaitForKRCTick();

        float alpha = float(i + 1) / steps;
        commandPose[axisIndex] = startPose[axisIndex] + alpha * delta;
        fri.SetCommandedCartPose(commandPose);

        //  LOG CSV 
        static std::ofstream csv("robot_pose.csv", std::ios::app);
        csv << now() << ","
            << commandPose[3] << ","
            << commandPose[7] << ","
            << commandPose[11] << "\n";
    }

    //  WYŚLIJ "FINISHED" 
    int finished = 2;
    send(client, (char*)&finished, sizeof(int), 0);
    csv_cmd << now() << ",FINISHED," << axisIndex << "," << delta << "\n";
    csv_cmd.flush();

    robotState = RobotState::IDLE;
    return true;
}


int main()
{
    const char* initFile = "980039-FRI-Driver.txt";

    try
    {
        // --- TCP SETUP ---
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);

        SOCKET server = socket(AF_INET, SOCK_STREAM, 0);

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(5000);
        addr.sin_addr.s_addr = INADDR_ANY;

        bind(server, (sockaddr*)&addr, sizeof(addr));
        listen(server, 1);

        std::cout << "[C++] time=" << now() << "Waiting for Python connection...\n";
        SOCKET client = accept(server, NULL, NULL);
        std::cout << "[C++] time=" << now() << "Python connected.\n";

        // --- FRI SETUP ---
        FastResearchInterface fri(initFile);
        fri.StartRobot(FastResearchInterface::CART_IMPEDANCE_CONTROL);
        setInitialStiffness(fri);

        std::cout << "[C++] time=" << now() << "Robot ready. Waiting for commands...\n";

        // --- MAIN LOOP ---
        while (true)
        {
            fri.WaitForKRCTick();

            int axis;
            float delta;

            if (receiveCommand(client, axis, delta))
            {
                if (robotState == RobotState::MOVING)
                {
                    int rejected = -1;
                    send(client, (char*)&rejected, sizeof(int), 0);
                    std::cout << "[C++] REJECTED command (robot busy)\n";
                    continue;
                }
                std::cout << "[C++] time=" << now()<< "  ACCEPTED axis=" << axis
                    << " delta=" << delta
                    << "\n";

                moveAxis(fri, axis, delta, client);
            }
        }



        fri.StopRobot();
        closesocket(client);
        closesocket(server);
        WSACleanup();
    }
    catch (std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
