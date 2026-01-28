clear; close all; clc;

filename = 'tylko-bonemarkery.csv';
frameRate = 50;
videoOutFile = 'markery_3D_animacja1.mp4';  

raw = readmatrix(filename);
raw(isinf(raw)) = NaN;
raw(raw == 0) = NaN;

nFrames = size(raw, 1);
dataOnly = raw(:, 3:end);
nMarkers = floor(size(dataOnly, 2) / 3);

Data.nFrames = nFrames;
Data.nMarkers = nMarkers;
Data.markers = NaN(nMarkers, 3, nFrames);

for f = 1:nFrames
    frameData = dataOnly(f, :);
    for m = 1:nMarkers
        idx = (m - 1) * 3 + 1;
        Data.markers(m, :, f) = frameData(idx:idx+2);
    end
end

fig = figure;
M0 = squeeze(Data.markers(:,:,1));
scatterObj = scatter3(M0(:,1), M0(:,2), M0(:,3), 50, 'filled');
labelHandles = gobjects(nMarkers,1);
hold on;

for m = 1:nMarkers
    labelHandles(m) = text(M0(m,1), M0(m,2), M0(m,3), ...
        num2str(m), 'VerticalAlignment', 'bottom', 'FontSize', 8);
end

grid on;
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
axis equal;
view(3);

v = VideoWriter(videoOutFile, 'MPEG-4');
v.FrameRate = frameRate;
open(v);

for f = 1:Data.nFrames
    M = squeeze(Data.markers(:,:,f));

    set(scatterObj, 'XData', M(:,1), 'YData', M(:,2), 'ZData', M(:,3));
    
    for m = 1:nMarkers
        set(labelHandles(m), 'Position', M(m,:));
    end
    
    title(sprintf('Klatka %d / %d', f, Data.nFrames));
    drawnow;

    frame = getframe(fig);
    writeVideo(v, frame);
end

close(v);
fprintf('Zapisano animację do pliku: %s\n', videoOutFile);
