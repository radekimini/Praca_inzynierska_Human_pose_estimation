clc; clear; close all;

% === Configuration ===
csv_file = 'OptitrackAgataPassiveWAggressive3.csv';   % OptiTrack CSV file
video_file = 'Agata_nagranie_z_pointem_na_kamerze3.mp4';% Input video file
output_video_file = 'annotated_output_2.mp4'; % Output video file
scale = 329;                              % Scale: meters → pixels
offset = [200, 60];                       % Offset: (X,Y) in pixels
video_fps = 30;
optitrack_fps = 120;
frame_ratio = optitrack_fps / video_fps;
start_video_frame = 1;
total_output_frames = 300;                % Programmer-defined number of frames

% === Read video ===
video = VideoReader(video_file);

% === Read and process CSV ===
raw = readmatrix(csv_file);
raw(raw == 0 | isinf(raw)) = NaN;

% === Build full skeleton array ===
% szkielet Radek_bez W
% skeleton = zeros(size(raw,1), 50);
% skeleton(:,3:5)  = raw(:,219:221); % 1
% skeleton(:,6:8)  = raw(:,207:209); % 2
% skeleton(:,9:11) = raw(:,195:197); % 3
% skeleton(:,12:14)= raw(:,159:161); % 4
% skeleton(:,15:17)= raw(:,171:173); % 5
% skeleton(:,18:20)= raw(:,183:185); % 6
% skeleton(:,21:23)= (raw(:,195:197) + raw(:,159:161)) / 2; % 7
% skeleton(:,24:26)= (((raw(:,117:119) + raw(:,120:122))/2) + ((raw(:,72:74) + raw(:,75:77))/2))/2; % 8
% skeleton(:,27:29)= raw(:,54:56);   % 9
% skeleton(:,30:32)= raw(:,57:59);   %10
% skeleton(:,33:35)= raw(:,144:146); %11
% skeleton(:,36:38)= (raw(:,129:131) + raw(:,132:134))/2; %12
% skeleton(:,39:41)= (raw(:,117:119) + raw(:,120:122))/2; %13
% skeleton(:,42:44)= (raw(:,72:74) + raw(:,75:77))/2;     %14
% skeleton(:,45:47)= (raw(:,84:86) + raw(:,87:89))/2;     %15
% skeleton(:,48:50)= raw(:,99:101);  %16

%szkielet Agata _ z W

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
skeleton(:,33:35) = raw(7:end,163:165);%11
skeleton(:,36:38) = (raw(7:end,140:142) + raw(7:end,143:145))/2;%12 
skeleton(:,39:41) = (raw(7:end,127:129) + raw(7:end,130:132))/2;%13
skeleton(:,42:44) = (raw(7:end,78:80) + raw(7:end,81:83))/2;%14
skeleton(:,45:47) = (raw(7:end,91:93) + raw(7:end,94:96))/2;%15
skeleton(:,48:50) = raw(7:end,114:116);%16
skeleton(:,51:53) = raw(7:end,372:374); % punkt stały 17
skeleton(:,54:56) = raw(7:end,375:377); % punkt stały 18
skeleton(:,57:59) = raw(7:end,378:380); % punkt stały 19
skeleton(:,60:62) = raw(7:end,381:383); % kamera

% === Create VideoWriter ===
% output_video = VideoWriter(output_video_file);
output_video = VideoWriter(output_video_file, 'MPEG-4');

output_video.FrameRate = video_fps;
open(output_video);

% === Plot and Save ===
fig = figure('Name', 'Annotated Video (Close to Stop Early)');

for frame_idx = 1 : total_output_frames
    % Early exit if figure closed
    if ~isvalid(fig)
        disp('Figure closed by user. Saving video and exiting early...');
        break;
    end

    video_frame = start_video_frame + frame_idx - 1;
    optitrack_frame = round(video_frame * frame_ratio);

    % Break if data runs out
    if optitrack_frame > size(skeleton,1)
        disp('No more skeleton data. Exiting...');
        break;
    end

    % Read frame from video
    video.CurrentTime = (video_frame - 1) / video.FrameRate;
    img = readFrame(video);

    % Prepare joint positions
    X = skeleton(optitrack_frame, 3:3:56);
    Y = skeleton(optitrack_frame, 4:3:56);
    points2D = [X; Y]';
    points2D = points2D * scale + offset;

    % Flip Y axis
    img_height = size(img,1);
    points2D(:,2) = img_height - points2D(:,2);

    imshow(img); hold on;

    for i = 1:size(points2D,1)
        if any(isnan(points2D(i,:))), continue; end
        plot(points2D(i,1), points2D(i,2), 'ro', 'MarkerSize', 6, 'LineWidth', 1.5);
        text(points2D(i,1)+5, points2D(i,2), num2str(i), 'Color', 'yellow', 'FontSize', 8);
    end

    % Draw skeleton lines
    skeletonLines = [
        1 2; 2 3; 3 7; 4 7; 4 5; 5 6;
        7 8; 8 9; 9 10;
        8 13; 13 12; 12 11;
        8 14; 14 15; 15 16
    ];

    for i = 1:size(skeletonLines, 1)
        idx1 = skeletonLines(i,1);
        idx2 = skeletonLines(i,2);
        p1 = points2D(idx1, :);
        p2 = points2D(idx2, :);
        if any(isnan(p1)) || any(isnan(p2)), continue; end
        plot([p1(1), p2(1)], [p1(2), p2(2)], 'g-', 'LineWidth', 2);
    end

    title(sprintf('Output Frame %d | Video Frame %d | OptiTrack Frame %d', frame_idx, video_frame, optitrack_frame));
    drawnow;

    % Save frame from figure
    frame = getframe(gca); 
    writeVideo(output_video, frame);

    hold off;
end

% Finalize
if isvalid(fig)
    close(fig);
end
close(output_video);
disp('Annotated video saved successfully.');

