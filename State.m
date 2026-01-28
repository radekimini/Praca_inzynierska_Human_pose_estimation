clc;
clear;
close all;
filename = 'cpp_commands.csv';

opts = detectImportOptions(filename, ...
    'Delimiter', {'\t', ','}, ...
    'ReadVariableNames', false, ...
    'TextType', 'string');   

data = readtable(filename, opts);

time   = data{:,1};        
status = string(data{:,2});
axisID = data{:,3};
value  = data{:,4};

idx = status == "ACCEPTED";

time   = time(idx);
axisID = axisID(idx);
value  = value(idx);

time = string(time);  
t = datetime(time, 'InputFormat', 'HH:mm:ss.SSS');
t_sec = seconds(t - t(1));

N = length(t_sec);

stateX = zeros(N,1);
stateY = zeros(N,1);
stateZ = zeros(N,1);

for i = 1:N
    dir = sign(value(i));  % +1 / -1 / 0
    
    switch axisID(i)
        case 3      % X
            stateX(i) = dir;
        case 7      % Y
            stateY(i) = dir;
        case 11     % Z
            stateZ(i) = dir;
    end
end

figure('Name','Stany ruchu osi robota KUKA');

subplot(3,1,1)
stairs(t_sec, stateX, 'LineWidth', 1.5);
ylim([-1.5 1.5]);
yticks([-1 0 1]);
ylabel('X');
title('Stany ruchu osi (kierunek / czas)');
grid on;

subplot(3,1,2)
stairs(t_sec, stateY, 'LineWidth', 1.5);
ylim([-1.5 1.5]);
yticks([-1 0 1]);
ylabel('Y');
grid on;

subplot(3,1,3)
stairs(t_sec, stateZ, 'LineWidth', 1.5);
ylim([-1.5 1.5]);
yticks([-1 0 1]);
ylabel('Z');
xlabel('Czas [s]');
grid on;
