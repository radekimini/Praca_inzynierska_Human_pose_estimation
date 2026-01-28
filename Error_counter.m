clc; clear;

%% === 1. Konfiguracja ===
csv_file = 'Radek5_take5_rece.csv';
video_file = 'take5_rece_calibrated62.mp4';
video_fps = 25;
optitrack_fps = 120;
frame_ratio = optitrack_fps / video_fps;
start_video_frame = 50;
total_output_frames = 570;

video = VideoReader(video_file);

%% === 2. Dane kamery ===
camPos = [2.387222,	1.176419,	3.290892];
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

skeleton(:,51:53) = raw(7:end,372:374); % punkt stały 17
skeleton(:,54:56) = raw(7:end,381:383); % punkt stały 18
skeleton(:,57:59) = raw(7:end,378:380); % punkt stały 19

N = size(skeleton, 1); 

MW = cell(1,19);
MW_noT = cell(1,19);    
pW_cartesian = cell(1,19);
p2D = cell(1,19);

for i = 1:19
    scale = 2.2;
    theta = 0; 

    if i == 1
        translation = [-1.75, -1, 0];
    elseif i == 2
        translation = [-1.75, -1, 0];
    elseif i == 3
        translation = [-1.75, -1, 0];
    elseif i == 4
        translation = [-1.75, -1, 0];
    elseif i == 5
        translation = [-1.75, -1, 0];
    elseif i == 6
        translation = [-1.75, -1, 0];
    elseif i == 7
        translation = [-1.75, -1, 0];
    elseif i == 8
        translation = [-1.75, -0.9, 0];
    elseif i == 9
        translation = [-1.75, -0.9, 0];
    elseif i == 10
        translation = [-1.75, -1, 0];
    elseif i == 11
        translation = [-1.7, -0.85, -0.025];
    elseif i == 12
        translation = [-1.75, -0.85, 0];
    elseif i == 13
        translation = [-1.75, -0.9, 0];
    elseif i == 14
        translation = [-1.75, -0.9, 0];
    elseif i == 15
        translation = [-1.55, -0.8, 0];
    elseif i == 16
        translation = [-1.6, -0.85, 0.1];
    elseif i == 17
        translation = [-1.95, -1.2, 0];
    elseif i == 18
        translation = [-2.1, -1.3, 0];
    elseif i == 19
        translation = [-1.75, -0.9, 0];
    end

    theta_rad = deg2rad(theta);

    S = eye(4);
    S(1,1) = scale;
    S(2,2) = scale;
    S(3,3) = scale;

    Ry = eye(4);
    Ry(1,1) = cos(theta_rad);
    Ry(1,3) = sin(theta_rad);
    Ry(3,1) = -sin(theta_rad);
    Ry(3,3) = cos(theta_rad);

    T = eye(4);
    T(1:3,4) = translation(:);
    
    MW{i} = T * Ry * S;

    MW_noT{i} = Ry * S;

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

outputVideo = VideoWriter('skeleton_overlay.mp4','MPEG-4');
outputVideo.FrameRate = video_fps;
open(outputVideo);

figure;

errors_per_frame = NaN(19, total_output_frames);

for vf = 1:total_output_frames
    frame_idx = round((vf - 1) * frame_ratio) + start_video_frame;
    if frame_idx > N, break; end

    frame = read(video, vf);
    imshow(frame); hold on;

    points2D = zeros(19, 2);
    points2D_noT = zeros(19, 2); 

    for i = 1:19
        p = skeleton(frame_idx, (3*i):(3*i+2));  
        if any(isnan(p))
            points2D(i,:) = [NaN, NaN];
            points2D_noT(i,:) = [NaN, NaN];
            continue;
        end

        pL_h = [p, 1]';                          

        pW = MW{i} * pL_h;                       
        pC = MV * pW;                            
        pProj = MP * pC;                         
        pt2D = pProj(1:2) / pProj(4);            

        pW_noT = MW_noT{i} * pL_h;               
        pC_noT = MV * pW_noT;                    
        pProj_noT = MP * pC_noT;                 
        pt2D_noT = pProj_noT(1:2) / pProj_noT(4);

        img_width = size(frame, 2);
        img_height = size(frame, 1);

        pt2D(1) = (pt2D(1) + 1) / 2 * img_width;
        pt2D(2) = (1 - (pt2D(2) + 1) / 2) * img_height;  

        pt2D_noT(1) = (pt2D_noT(1) + 1) / 2 * img_width;
        pt2D_noT(2) = (1 - (pt2D_noT(2) + 1) / 2) * img_height;  

        points2D(i,:) = pt2D';
        points2D_noT(i,:) = pt2D_noT';

        plot(pt2D(1), pt2D(2), 'ro', 'MarkerSize', 6, 'LineWidth', 1.5);
        text(pt2D(1)+5, pt2D(2), sprintf('%d', i), 'Color', 'y', 'FontSize', 8);

        err = sqrt( (pt2D(1) - pt2D_noT(1))^2 + (pt2D(2) - pt2D_noT(2))^2 );
        errors_per_frame(i, vf) = err/100;
    end

    for i = 1:size(skeletonLines, 1)
        idx1 = skeletonLines(i,1);
        idx2 = skeletonLines(i,2);
        p1 = points2D(idx1, :);
        p2 = points2D(idx2, :);
        if any(isnan(p1)) || any(isnan(p2)), continue; end
        plot([p1(1), p2(1)], [p1(2), p2(2)], 'g-', 'LineWidth', 2);
    end

    frame_errs = errors_per_frame(:, vf);
    frame_mean_err = mean(frame_errs(~isnan(frame_errs)));
    title(sprintf('Klatka: %d (CSV %d)   |   Mean pixel error (this frame): %.2f px', vf, frame_idx, frame_mean_err));
    pause(1/video_fps);

    frame_out = getframe(gcf);
    writeVideo(outputVideo, frame_out);

    hold off;
end

close(outputVideo);
disp('Wideo zapisane jako skeleton_overlay.mp4');

%% === 6. Analiza błędów ===
mean_err_per_point = nanmean(errors_per_frame, 2);   
median_err_per_point = nanmedian(errors_per_frame, 2);
std_err_per_point = nanstd(errors_per_frame, 0, 2);
count_per_point = sum(~isnan(errors_per_frame), 2);

all_errors_vec = errors_per_frame(~isnan(errors_per_frame));
overall_RMSE = sqrt(mean(all_errors_vec.^2));

overall_mean = mean(all_errors_vec);
overall_median = median(all_errors_vec);

fprintf('\n--- Overlay error summary ---\n');
for i = 1:19
    fprintf('Point %2d: mean=%.3f px, median=%.3f px, std=%.3f px, samples=%d\n', ...
        i, mean_err_per_point(i), median_err_per_point(i), std_err_per_point(i), count_per_point(i));
end
fprintf('Overall RMSE (all points, all frames): %.3f px\n', overall_RMSE);
fprintf('Overall mean error: %.3f px, overall median error: %.3f px\n', overall_mean, overall_median);

save('overlay_errors.mat', 'errors_per_frame', 'mean_err_per_point', 'median_err_per_point', ...
    'std_err_per_point', 'count_per_point', 'overall_RMSE', 'overall_mean', 'overall_median');

disp('Błędy zapisane w overlay_errors.mat');

%% === 7. Wizualizacja błędów ===
figure('Name', 'Analiza błędów', 'Position', [100 100 1200 800]);

%% (i) Wykres średniego błędu per-point
subplot(2,2,1);
bar(1:19, mean_err_per_point, 'FaceColor', [0.2 0.6 0.9]);
hold on;
errorbar(1:19, mean_err_per_point, std_err_per_point, 'k.', 'LineWidth', 1);
xlabel('Numer punktu');
ylabel('Średni błąd [px]');
title('Średni błąd i odchylenie standardowe (per punkt)');
grid on;
xlim([0.5 19.5]);

%% (ii) Mapa błędów w czasie (heatmapa)
subplot(2,2,2);
imagesc(errors_per_frame);
colormap('hot');
colorbar;
xlabel('Numer klatki');
ylabel('Numer punktu');
title('Mapa błędów w czasie (px)');
set(gca, 'YDir', 'normal');
yticks(1:19);
xticks(1:50:total_output_frames);

%% (iii) Histogram błędów
subplot(2,2,[3 4]);
all_errs = errors_per_frame(~isnan(errors_per_frame));
histogram(all_errs, 40, 'FaceColor', [0.3 0.8 0.4], 'EdgeColor', 'k');
xlabel('Błąd [px]');
ylabel('Liczba wystąpień');
title(sprintf('Histogram błędów projekcji (N=%d) | RMSE=%.2f px', numel(all_errs), overall_RMSE));
grid on;

sgtitle('Analiza błędów translacji (projekcja 2D)', 'FontSize', 14, 'FontWeight', 'bold');

disp('Wygenerowano wykresy błędów: średnie, mapa i histogram.');
