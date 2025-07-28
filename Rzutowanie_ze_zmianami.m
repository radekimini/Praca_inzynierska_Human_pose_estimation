clc; clear;

%% === 1. Konfiguracja ===
csv_file = 'OptitrackAgataPassiveWAggressive3.csv';
video_file = 'Agata_nagranie_z_pointem_na_kamerze3.mp4';
video_fps = 30;
optitrack_fps = 120;
frame_ratio = optitrack_fps / video_fps;
start_video_frame = 1;
total_output_frames = 10;

video = VideoReader(video_file);

%% === 2. Dane kamery ===
camPos = [2.388, 1.25, 3.60];
lookDir = [0, 0, 1];  % poprawny kierunek patrzenia
upVec = [0, 1, 0];

normalizeVec = @(v) v / norm(v + 1e-8);  % zapobiega dzieleniu przez zero

camZ = normalizeVec(lookDir(:));
camX = normalizeVec(cross(upVec(:), camZ));
camY = normalizeVec(cross(camZ, camX));

R = [camX'; camY'; camZ'];
t = -R * camPos(:);

MV = eye(4);
MV(1:3,1:3) = R;
MV(1:3,4) = t;

fovY = deg2rad(82);
aspect = 16 / 9;
zN = 2.07;
zF = 4.10;

tang = tan(fovY / 2);
S_y = 1 / tang;
S_x = S_y / aspect;

MP = zeros(4);
MP(1,1) = S_x;
MP(2,2) = S_y;
MP(3,3) = -(zF + zN)/(zF - zN);
MP(3,4) = -2 * zF * zN / (zF - zN);
MP(4,3) = -1;

%% === 3. Wczytaj dane z CSV ===
raw = readmatrix(csv_file);
raw(raw == 0 | isinf(raw)) = NaN;

skeleton(:,1:2) = raw(7:end,1:2);
skeleton(:,3:5) = raw(7:end,238:240); %1
skeleton(:,6:8) = raw(7:end,225:227); %2
skeleton(:,9:11) = raw(7:end,212:214);%3
skeleton(:,12:14) = raw(7:end,173:175);%4
skeleton(:,15:17) = raw(7:end,186:188);%5
skeleton(:,18:20) = raw(7:end,199:201);%6
skeleton(:,21:23) = (raw(7:end,212:214) + raw(7:end,173:175))/2;%7
skeleton(:,24:26) = (((raw(7:end,127:129) + raw(7:end,130:132))/2) + ((raw(7:end,78:80) + raw(7:end,81:83))/2))/2;%8
skeleton(:,27:29) = raw(7:end,59:61);%9
skeleton(:,30:32) = raw(7:end,62:64);%10
skeleton(:,33:35) = raw(7:end,157:159);%11
skeleton(:,36:38) = (raw(7:end,140:142) + raw(7:end,143:145))/2;%12 
skeleton(:,39:41) = (raw(7:end,127:129) + raw(7:end,130:132))/2;%13
skeleton(:,42:44) = (raw(7:end,78:80) + raw(7:end,81:83))/2;%14
skeleton(:,45:47) = (raw(7:end,91:93) + raw(7:end,94:96))/2;%15
skeleton(:,48:50) = raw(7:end,108:110);%16

% Dodatkowe 3 markery
skeleton(:,51:53) = raw(7:end,372:374); % punkt stały 17
skeleton(:,54:56) = raw(7:end,375:377); % punkt stały 18
skeleton(:,57:59) = raw(7:end,378:380); % punkt stały 19

N = size(skeleton, 1); % liczba klatek

MW = cell(1,19);
pW_cartesian = cell(1,19);
p2D = cell(1,19);

%Wartości dla wszystkich danych
for i = 1:19
    scale = 1;
    theta = 0; % w stopniach
    translation = [0, 0, 0];

    %indywidualne wartości dla każdego punktu
    if i == 1
        scale = 1.25;
        theta = 0;
        translation = [0.1, -0.75, 0];
    elseif i == 2
        scale = 1.25;
        theta = 0;
        translation = [0.05, -0.4, 0];
    elseif i == 3
        scale = 1.25;
        theta = 0;
        translation = [0, 0, 0];
    elseif i == 4
        scale = 1.25;
        theta = 0;
        translation = [0.3, 0, 0];
    elseif i == 5
        scale = 1.25;
        theta = 0;
        translation = [0.3, -0.4, 0];
        elseif i == 6
        scale = 1.25;
        theta = 0;
        translation = [0.32, -0.79, 0];
    elseif i == 7
        scale = 1.25;
        theta = 0;
        translation = [0.15, 0, 0];
    elseif i == 8
        scale = 1.25;
        theta = 0;
        translation = [0.15, 0.55, 0];
    elseif i == 9
        scale = 1.25;
        theta = 0;
        translation = [0.15, 0.65, 0];
    elseif i == 10
        scale = 1.25;
        theta = 0;
        translation = [0.15, 0.9, 0];
    elseif i == 11
        scale = 1.25;
        theta = 0;
        translation = [-0.6, 0.55, 0];
    elseif i == 12
        scale = 1.25;
        theta = 0;
        translation = [-0.3, 0.55, 0];
    elseif i == 13
        scale = 1.25;
        theta = 0;
        translation = [-0, 0.55, 0];
    elseif i == 14
        scale = 1.25;
        theta = 0;
        translation = [0.3, 0.55, 0];
    elseif i == 15
        scale = 1.25;
        theta = 0;
        translation = [0.7, 0.58, 0];
    elseif i == 16
        scale = 1.25;
        theta = 0;
        translation = [1, 0.67, 0];
    elseif i == 17
        scale = 1;
        theta = 0;
        translation = [-0.6, -0.9, 0];
    elseif i == 18
        scale = 1;
        theta = 0;
        translation = [-0.77, -0.88, 0];
    elseif i == 19
        scale = 1;
        theta = 0;
        translation = [-0.77, -0.88, 0];
    end

    % Konwersja kąta
    theta_rad = deg2rad(theta);

    % Skalowanie
    S = eye(4);
    S(1,1) = scale;
    S(2,2) = scale;
    S(3,3) = scale;

    % Rotacja wokół osi Y
    Ry = eye(4);
    Ry(1,1) = cos(theta_rad);
    Ry(1,3) = sin(theta_rad);
    Ry(3,1) = -sin(theta_rad);
    Ry(3,3) = cos(theta_rad);

    % Translacja
    T = eye(4);
    T(1:3,4) = translation(:);

    % Macierz transformacji dla danego punktu
    MW{i} = T * Ry * S;

    % Transformacja punktów
    p = skeleton(:, (3*i):(3*i+2));   % punkty XYZ
    pL_h = [p, ones(N,1)];           % punkty homogeniczne Nx4

    pW = (MW{i} * pL_h')';           % Nx4
    pW_cartesian{i} = pW(:,1:3);     % XYZ

    % Projekcja do przestrzeni kamery
    pC = (MV * [pW_cartesian{i}, ones(N,1)]')';    % Nx4

    % Projekcja do przestrzeni ekranowej (homogenicznej)
    pProj = (MP * pC')';                          % Nx4
    p2D{i} = pProj(:,1:2) ./ pProj(:,4);          % dzielimy przez W (perspektywa)
end

%% === 5. Rysowanie ===
skeletonLines = [
    1 2; 2 3; 3 7; 4 7; 4 5; 5 6;
    7 8; 8 9; 9 10;
    8 13; 13 12; 12 11;
    8 14; 14 15; 15 16
];

figure;
for vf = 1:total_output_frames
    frame_idx = round((vf - 1) * frame_ratio) + start_video_frame;
    if frame_idx > N, break; end

    % Czytaj ramkę wideo
    frame = read(video, vf);
    imshow(frame); hold on;

    % Dla każdej ramki – wyciągnij dane 3D i przekształć dynamicznie
    points2D = zeros(19, 2);

    for i = 1:19
        p = skeleton(frame_idx, (3*i):(3*i+2));  % 1x3 punkt
        if any(isnan(p)), points2D(i,:) = [NaN, NaN]; continue; end

        pL_h = [p, 1]';                          % 4x1

        % Transformacja świata → kamery → projekcja
        pW = MW{i} * pL_h;                       % 4x1
        pC = MV * pW;                            % 4x1
        pProj = MP * pC;                         % 4x1
        pt2D = pProj(1:2) / pProj(4);            % 2D współrzędne
        img_width = size(frame, 2);
img_height = size(frame, 1);

% Skalowanie z NDC [-1, 1] do [0, width] i [0, height]
pt2D(1) = (pt2D(1) + 1) / 2 * img_width;
pt2D(2) = (1 - (pt2D(2) + 1) / 2) * img_height;  % flip Y


        points2D(i,:) = pt2D';

        % Rysowanie punktu
        plot(pt2D(1), pt2D(2), 'ro', 'MarkerSize', 6, 'LineWidth', 1.5);
        text(pt2D(1)+5, pt2D(2), sprintf('%d', i), 'Color', 'y', 'FontSize', 8);
    end

    % Rysowanie połączeń
    for i = 1:size(skeletonLines, 1)
        idx1 = skeletonLines(i,1);
        idx2 = skeletonLines(i,2);
        p1 = points2D(idx1, :);
        p2 = points2D(idx2, :);
        if any(isnan(p1)) || any(isnan(p2)), continue; end
        plot([p1(1), p2(1)], [p1(2), p2(2)], 'g-', 'LineWidth', 2);
    end

    title(sprintf('Klatka: %d (CSV %d)', vf, frame_idx));
    pause(1/video_fps);
    hold off;
end