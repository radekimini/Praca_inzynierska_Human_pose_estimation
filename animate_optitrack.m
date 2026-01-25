close all; clc;

filename = 'Take 2025-07-28 02.37.20 PM.csv';
frameRate = 100;

raw = readmatrix(filename);
raw(isinf(raw)) = NaN;
raw(raw == 0) = NaN;
size_skeleton = size(raw);
skeleton = zeros(size_skeleton(1) - 6, 65);

%%% szkielet agata
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
skeleton(:,54:56) = raw(7:end,375:377); % punkt stały 18
skeleton(:,57:59) = raw(7:end,378:380); % punkt stały 19
skeleton(:,60:62) = raw(7:end,366:368); % kamera

% Dodajemy punkt 21 (nr 21) na stałe: [0,0,0]
fixedPoint21 = repmat([0,0,0], size(skeleton,1), 1);
skeleton(:,51:53) = fixedPoint21;

% Liczba klatek i markerów
nFrames = size(skeleton, 1);
dataOnly = skeleton(:, 3:end);
nMarkers = floor(size(dataOnly, 2) / 3);

% Struktura danych
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

% Wierzchołki workspace: 101–108
workspaceVertices = [
    0, 0, -0.5;     % 101
    0, 0, 3.6;       % 102
    4, 0, 3.6;       % 103
    4, 2.5, 3.6;      % 104
    0, 2.5, 3.6;     % 105
    4, 0, -0.5;      % 106
    0, 2.5, -0.5;    % 107
    4, 2.5, -0.5    % 108
    0, 0, 1.53      %109
    4, 0, 1.53      %110
    0, 2.5, 1.53    %111
    4, 2.5, 1.53    %112
];

% Krawędzie workspace
workspaceEdges = [
    101 102; 102 103; 103 104; 104 105; 105 102;         
    107 108; 108 106;  
    101 106; 101 107; 103 106; 104 108; 105 107  ;
    109 110; 110 112; 111 112; 111 109;
];

figure; hold on;

% Pierwsza klatka szkieletu
M0 = squeeze(Data.markers(:,:,1));
scatterObj = scatter3(M0(:,1), M0(:,2), M0(:,3), 50, 'filled');

% Linie szkieletu
skeletonLines = [
    1 2; 2 3; 3 7; 4 7; 4 5; 5 6;
    7 8; 8 9; 9 10;
    8 13; 13 12; 12 11;
    8 14; 14 15; 15 16
];

lineHandles = gobjects(size(skeletonLines,1),1);
for i = 1:size(skeletonLines,1)
    idx1 = skeletonLines(i,1);
    idx2 = skeletonLines(i,2);
    lineHandles(i) = plot3([M0(idx1,1) M0(idx2,1)], ...
                           [M0(idx1,2) M0(idx2,2)], ...
                           [M0(idx1,3) M0(idx2,3)], ...
                           'g-', 'LineWidth', 2);
end

% Etykiety markerów
labelHandles = gobjects(nMarkers,1);
for m = 1:nMarkers
    labelHandles(m) = text(M0(m,1), M0(m,2), M0(m,3), ...
        num2str(m), 'VerticalAlignment', 'bottom', 'FontSize', 8);
end

% Wierzchołki workspace
scatter3(workspaceVertices(:,1), workspaceVertices(:,2), workspaceVertices(:,3), ...
    60, [0 0.4 0], 'filled');

% Etykiety workspace: 101–108
for i = 1:size(workspaceVertices,1)
    text(workspaceVertices(i,1), workspaceVertices(i,2), workspaceVertices(i,3), ...
        num2str(100+i), 'Color', [0 0.4 0], 'FontSize', 8);
end

% Linie workspace
for i = 1:size(workspaceEdges,1)
    idx1 = workspaceEdges(i,1)-100; 
    idx2 = workspaceEdges(i,2)-100;
    plot3([workspaceVertices(idx1,1), workspaceVertices(idx2,1)], ...
          [workspaceVertices(idx1,2), workspaceVertices(idx2,2)], ...
          [workspaceVertices(idx1,3), workspaceVertices(idx2,3)], ...
          'Color',[0 0.4 0],'LineWidth',2);
end

% --- DODANE: promienie z punktu kamery (nr 20)
cameraIdx = 20;  
horizontalFOV = deg2rad(82);
verticalFOV = deg2rad(52);
rayLength = 5.5;
leftRayDir = [ -sin(horizontalFOV/2), 0, -cos(horizontalFOV/2) ];
rightRayDir = [ sin(horizontalFOV/2), 0, -cos(horizontalFOV/2) ];
topRayDir = [ 0, sin(verticalFOV/2), -cos(verticalFOV/2) ];
bottomRayDir = [ 0, -sin(verticalFOV/2), -cos(verticalFOV/2) ];

violet = [0.5 0 0.5];

cameraPoint = M0(cameraIdx,:);
rayHandles(1) = plot3([cameraPoint(1), cameraPoint(1)+rayLength*leftRayDir(1)], ...
    [cameraPoint(2), cameraPoint(2)+rayLength*leftRayDir(2)], ...
    [cameraPoint(3), cameraPoint(3)+rayLength*leftRayDir(3)], 'Color', violet, 'LineWidth', 2);
rayHandles(2) = plot3([cameraPoint(1), cameraPoint(1)+rayLength*rightRayDir(1)], ...
    [cameraPoint(2), cameraPoint(2)+rayLength*rightRayDir(2)], ...
    [cameraPoint(3), cameraPoint(3)+rayLength*rightRayDir(3)], 'Color', violet, 'LineWidth', 2);
rayHandles(3) = plot3([cameraPoint(1), cameraPoint(1)+rayLength*topRayDir(1)], ...
    [cameraPoint(2), cameraPoint(2)+rayLength*topRayDir(2)], ...
    [cameraPoint(3), cameraPoint(3)+rayLength*topRayDir(3)], 'Color', violet, 'LineWidth', 2);
rayHandles(4) = plot3([cameraPoint(1), cameraPoint(1)+rayLength*bottomRayDir(1)], ...
    [cameraPoint(2), cameraPoint(2)+rayLength*bottomRayDir(2)], ...
    [cameraPoint(3), cameraPoint(3)+rayLength*bottomRayDir(3)], 'Color', violet, 'LineWidth', 2);

grid on;
xlabel('X'); ylabel('Y'); zlabel('Z');
axis equal;

% Animacja
for f = 1:Data.nFrames
    M = squeeze(Data.markers(:,:,f));
    set(scatterObj, 'XData', M(:,1), 'YData', M(:,2), 'ZData', M(:,3));
    for m = 1:nMarkers
        set(labelHandles(m), 'Position', M(m,:));
    end
    for i = 1:size(skeletonLines,1)
        idx1 = skeletonLines(i,1);
        idx2 = skeletonLines(i,2);
        set(lineHandles(i), ...
            'XData', [M(idx1,1), M(idx2,1)], ...
            'YData', [M(idx1,2), M(idx2,2)], ...
            'ZData', [M(idx1,3), M(idx2,3)]);
    end
    
    % Aktualizuj promienie
    cameraPoint = M(cameraIdx,:);
    set(rayHandles(1), 'XData', [cameraPoint(1), cameraPoint(1)+rayLength*leftRayDir(1)], ...
        'YData', [cameraPoint(2), cameraPoint(2)+rayLength*leftRayDir(2)], ...
        'ZData', [cameraPoint(3), cameraPoint(3)+rayLength*leftRayDir(3)]);
    set(rayHandles(2), 'XData', [cameraPoint(1), cameraPoint(1)+rayLength*rightRayDir(1)], ...
        'YData', [cameraPoint(2), cameraPoint(2)+rayLength*rightRayDir(2)], ...
        'ZData', [cameraPoint(3), cameraPoint(3)+rayLength*rightRayDir(3)]);
    set(rayHandles(3), 'XData', [cameraPoint(1), cameraPoint(1)+rayLength*topRayDir(1)], ...
        'YData', [cameraPoint(2), cameraPoint(2)+rayLength*topRayDir(2)], ...
        'ZData', [cameraPoint(3), cameraPoint(3)+rayLength*topRayDir(3)]);
    set(rayHandles(4), 'XData', [cameraPoint(1), cameraPoint(1)+rayLength*bottomRayDir(1)], ...
        'YData', [cameraPoint(2), cameraPoint(2)+rayLength*bottomRayDir(2)], ...
        'ZData', [cameraPoint(3), cameraPoint(3)+rayLength*bottomRayDir(3)]);
    
    title(sprintf('Klatka %d / %d', f, Data.nFrames));
    drawnow limitrate;
end
