clc; clear;

% === Konfiguracja ===
csv_file = 'Radzio8_Casual.csv';
video_file = 'Take8_Calibrated10.mp4';
video_fps = 25; 
optitrack_fps = 120;
frame_ratio = optitrack_fps / video_fps;
start_video_frame = 1;
total_output_frames = 1000;
video = VideoReader(video_file);

%% === 2. Dane kamery ===
camPos = [2.387222, 1.176419, 3.290892];
lookDir = [0, 0, 1];
upVec = [0, 1, 0];

normalizeVec = @(v) v / norm(v + 1e-8);
camZ = normalizeVec(lookDir(:));
camX = normalizeVec(cross(upVec(:), camZ));
camY = normalizeVec(cross(camZ, camX));

R = [camX'; camY'; camZ'];
t = -R * camPos(:);

MV = eye(4);
MV(1:3,1:3) = R;
MV(1:3,4) = t;

fovY = deg2rad(90);
aspect = 4 / 3;
zN = 2.7;
zF = 3.29;

tang = tan(fovY / 2);
S_y = 1 / tang;
S_x = S_y / aspect;

MP = zeros(4);
MP(1,1) = S_x;
MP(2,2) = S_y;
MP(3,3) = (zF + zN)/(zF - zN);
MP(3,4) = 2 * zF * zN / (zF - zN);
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
%skeleton(:,51:53) = raw(7:end,372:374); % punkt stały 17
%skeleton(:,54:56) = raw(7:end,381:383); % punkt stały 18
skeleton(:,57:59) = raw(7:end,378:380); % punkt stały 19

N = size(skeleton, 1); % liczba klatek

MW = cell(1,19);
pW_cartesian = cell(1,19);
p2D = cell(1,19);

for i = 1:19
     % Domyślne wartości
     scale = 3.4;
     theta = 0; % w stopniach
%     translation = [-4.65, -2, 0];
 
    % Przypisz indywidualne wartości dla każdego punktu
    if i == 1
        translation = [-4.65, -2.2, 0];
    elseif i == 2
        translation = [-4.65, -2.2, 0];
    elseif i == 3
        translation = [-4.65, -2.2, 0];
    elseif i == 4
        translation = [-4.65, -2.2, 0];
    elseif i == 5
        translation = [-4.65, -2.2, 0];
    elseif i == 6
        translation = [-4.65, -2.2, 0];
    elseif i == 7
        translation = [-4.65, -2.2, 0];
    elseif i == 8
        translation = [-4.65, -2.2, 0];
    elseif i == 9
        translation = [-4.65, -2.2, 0];
    elseif i == 10
        translation = [-4.65, -2.2, 0];
    elseif i == 11
        translation = [-4.65, -2.2, 0];
    elseif i == 12
        translation = [-4.65, -2.2, 0];
    elseif i == 13
        translation = [-4.65, -2.2, 0];
    elseif i == 14
        translation = [-4.65, -2.2, 0];
    elseif i == 15
        translation = [-4.65, -1.9, -0.1];
    elseif i == 16
        translation = [-4.55, -1.9, 0];
    elseif i == 17
        translation = [-4.65, -2.2, 0];
    elseif i == 18
        translation = [-4.65, -2.2, 0];
    elseif i == 19
        translation = [-4.65, -2.2, 0];
    end

    theta_rad = deg2rad(theta);
    S = eye(4); S(1:3,1:3) = scale*eye(3);
    Ry = eye(4);
    Ry([1 3],[1 3]) = [cos(theta_rad) sin(theta_rad); -sin(theta_rad) cos(theta_rad)];
    T = eye(4); T(1:3,4) = translation(:);

    MW{i} = T * Ry * S;
    p = skeleton(:, (3*i):(3*i+2));
    pL_h = [p, ones(N,1)];
    pW = (MW{i} * pL_h')';
    pW_cartesian{i} = pW(:,1:3);

    pC = (MV * [pW_cartesian{i}, ones(N,1)]')';
    pProj = (MP * pC')';
    p2D{i} = pProj(:,1:2) ./ pProj(:,4);
end

%% === 5. Rysowanie ===
skeletonLines = [
    1 2; 2 3; 3 7; 4 7; 4 5; 5 6;
    7 8; 8 9; 9 10;
    8 13; 13 12; 12 11;
    8 14; 14 15; 15 16
];

pairs_vert = [2 3; 4 5; 12 13; 14 15];
joint_triplets = [1 2 3; 4 5 6; 11 12 13; 14 15 16];

right_side_idx = [1,3];
left_side_idx  = [2,4];

figure;
for vf = 1:total_output_frames
    frame_idx = round((vf-1)*frame_ratio) + start_video_frame;
    if frame_idx > size(skeleton,1), break; end
    frame = read(video, vf);
    imshow(frame); hold on;

    % Oś pionowa (lokalna "vert")
    p7  = pW_cartesian{7}(frame_idx,:);
    p10 = pW_cartesian{10}(frame_idx,:);
    v_vert = (p10 - p7)./norm(p10-p7);

    % --- znaki wg lokalnych układów w 12 i 15 ---
    y11 = pW_cartesian{11}(frame_idx,2);
    y12 = pW_cartesian{12}(frame_idx,2);
    y15 = pW_cartesian{15}(frame_idx,2);
    y16 = pW_cartesian{16}(frame_idx,2);

    sign12 = 1;  % dla 11-12-13 oraz segmentu 12-13 względem vert
    if ~(isnan(y11) || isnan(y12))
        if y11 < y12   % 11 jest niżej niż 12
            sign12 = -1;
        end
    end

    sign15 = 1;  % dla 14-15-16 oraz segmentu 14-15 względem vert
    if ~(isnan(y16) || isnan(y15))
        if y16 < y15   % 16 jest niżej niż 15
            sign15 = -1;
        end
    end
    % ---------------------------------------------

    % Kąty segmentów vs vert (z unifikacją i znakiem dla 12-13, 14-15)
    angles_vert = zeros(4,1);
    for k = 1:4
        a = pairs_vert(k,1); b = pairs_vert(k,2);
        u = pW_cartesian{b}(frame_idx,:) - pW_cartesian{a}(frame_idx,:);
        ang = atan2(norm(cross(u,v_vert)), dot(u,v_vert));  % [0..pi]
        ang = min(ang, pi - ang);                           % unifikacja 0..pi/2

        % znak wg lokalnego OY (tylko dla ramion)
        if a==12 && b==13
            ang = sign12 * ang;
        elseif a==14 && b==15
            ang = sign15 * ang;
        end
        angles_vert(k) = ang;
    end

    % Kąty stawów (z unifikacją i znakiem dla 11-12-13 oraz 14-15-16)
    angles_joint = zeros(4,1);
    for k = 1:4
        tri = joint_triplets(k,:);
        u1 = pW_cartesian{tri(2)}(frame_idx,:) - pW_cartesian{tri(1)}(frame_idx,:);
        u2 = pW_cartesian{tri(3)}(frame_idx,:) - pW_cartesian{tri(2)}(frame_idx,:);
        ang = atan2(norm(cross(u1,u2)), dot(u1,u2));  % [0..pi]
        ang = min(ang, pi - ang);                     % unifikacja 0..pi/2

        if isequal(tri,[11 12 13])
            ang = sign12 * ang;
        elseif isequal(tri,[14 15 16])
            ang = sign15 * ang;
        end
        angles_joint(k) = ang;
    end

    % Kąt głowy (pozostaje bez znaku; unifikacja do mniejszego)
    head_vec = pW_cartesian{10}(frame_idx,:) - pW_cartesian{9}(frame_idx,:);
    head_angle = atan2(norm(cross(head_vec, v_vert)), dot(head_vec, v_vert));
    head_angle = min(head_angle, pi - head_angle);

    % === Napisy w prawym dolnym rogu ===
    baseX = size(frame,2) - 300;
    baseY = size(frame,1) - 20;
    dy = 18;

    texts = strings(0);
    cols  = zeros(0,3);

    for k = 1:4
        txt1 = sprintf('Seg %d-%d vs vert: %.1f°', ...
            pairs_vert(k,1), pairs_vert(k,2), rad2deg(angles_vert(k)));
        txt2 = sprintf('Joint %d-%d-%d: %.1f°', ...
            joint_triplets(k,:), rad2deg(angles_joint(k)));

        if ismember(k,right_side_idx)
            col = [0 0 0.5];
        else
            col = [0 0.4 0];
        end

        texts(end+1) = txt1; cols(end+1,:) = col;
        texts(end+1) = txt2; cols(end+1,:) = col;
    end
    texts(end+1) = sprintf('Head angle: %.1f°', rad2deg(head_angle));
    cols(end+1,:) = [0.6 0 0.6]; % fioletowy

    % === Prostokąt tła (biały) ===
    rectW = 300; rectH = dy*length(texts)+10;
    rectangle('Position',[baseX-5, baseY-rectH, rectW, rectH], ...
        'FaceColor',[1 1 1 0.7],'EdgeColor','none');

    % Wypisywanie tekstów
    for n = 1:length(texts)
        text(baseX, baseY - dy*(n-1), texts(n), ...
            'Color',cols(n,:), 'FontSize',9, 'FontWeight','bold', ...
            'HorizontalAlignment','left');
    end

    % Skeleton (bez zmian)
    points2D = nan(19,2);
    for i = 1:19
        p = skeleton(frame_idx,(3*i):(3*i+2));
        if any(isnan(p)), continue; end
        pL = [p,1]'; pW = MW{i}*pL; pC = MV*pW; pProj = MP*pC;
        pt = pProj(1:2)/pProj(4);
        imgW = size(frame,2); imgH = size(frame,1);
        pt(1) = (pt(1)+1)/2*imgW;
        pt(2) = (1-(pt(2)+1)/2)*imgH;
        points2D(i,:) = pt';
        plot(pt(1),pt(2),'ro','MarkerSize',6,'LineWidth',1.5);
        text(pt(1)+5,pt(2),sprintf('%d',i),'Color','y','FontSize',8);
    end
    for e = 1:size(skeletonLines,1)
        i1 = skeletonLines(e,1); i2 = skeletonLines(e,2);
        p1 = points2D(i1,:); p2 = points2D(i2,:);
        if any(isnan(p1))||any(isnan(p2)), continue; end
        plot([p1(1),p2(1)],[p1(2),p2(2)],'g-','LineWidth',2);
    end

    title(sprintf('Frame %d (CSV %d)', vf, frame_idx));
    pause(1/video_fps); hold off;
end

