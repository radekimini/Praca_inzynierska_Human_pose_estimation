opts = detectImportOptions('robot_pose.csv');
opts = setvartype(opts, 1, 'datetime');

data = readtable('robot_pose.csv', opts);

t = seconds(data{:,1} - data{1,1});

x = data{:,2};
y = data{:,3};
z = data{:,4};

dt = diff(t);

dx = diff(x);
dy = diff(y);
dz = diff(z);

valid = dt > 0;

vx = dx(valid) ./ dt(valid);
vy = dy(valid) ./ dt(valid);
vz = dz(valid) ./ dt(valid);

v = sqrt(vx.^2 + vy.^2 + vz.^2);

t_v = t(1:end-1);
t_v = t_v(valid);

x_v = x(1:end-1);
y_v = y(1:end-1);
z_v = z(1:end-1);

x_v = x_v(valid);
y_v = y_v(valid);
z_v = z_v(valid);

%% --- Trajektoria kolorowana czasem ---
figure;
scatter3(x, y, z, 12, t, 'filled');
cb = colorbar;
cb.Label.String = 'Czas [s]';

grid on;
axis equal;
xlabel('X [m]');
ylabel('Y [m]');
zlabel('Z [m]');
title('Trajektoria robota z oznaczeniem czasu');
view(3);
% 
% %% --- Trajektoria kolorowana prędkością ---
% figure;
% scatter3(x_v, y_v, z_v, 12, v, 'filled');
% colorbar;
% grid on;
% axis equal;
% xlabel('X [m]');
% ylabel('Y [m]');
% zlabel('Z [m]');
% title('Trajektoria robota kolorowana prędkością');
% view(3);

figure;
plot(t_v, v, 'LineWidth', 1.5);
grid on;
xlabel('Czas [s]');
ylabel('Prędkość [m/s]');
title('Moduł prędkości robota');
