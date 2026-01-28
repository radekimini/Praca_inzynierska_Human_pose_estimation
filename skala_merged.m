
csvFile = 'OptitrackAgataPassiveWAggressive3.csv';
ptIdxCSV = [17,18,19];         % punkty z CSV
realDistCSV = [15,20];         % cm: 17-18 i 18-19
realDistImg = 15;              % cm: kliknięcie na obrazie

raw = readmatrix(csv_file);
raw(raw == 0 | isinf(raw)) = NaN;

frame = 1;
frame_row = frame + 6; 

x17 = raw(8, 372);
x18 = raw(8, 375);
z18 = raw(8, 377);
z19 = raw(8, 380);
 
d17_18_px = abs(x18 - x17); 
d18_19_px = abs(z19 - z18); 

d17_18_cm = 15;
d18_19_cm = 20;

cmPerPx_X = (d17_18_cm / d17_18_px);
cmPerPx_Z = (d18_19_cm / d18_19_px);
cmPerPx = mean([cmPerPx_X, cmPerPx_Z]);

fprintf('Skala z X: %.5f cm/piksel | Skala z Z: %.5f cm/piksel | Średnia: %.5f cm/piksel\n', ...
    cmPerPx_X, cmPerPx_Z, cmPerPx);

numImages = 15;
cmPerPx_Imgs = zeros(1, numImages);
figure;

for i = 0:numImages-1
    imgFile = sprintf('skala%d.png', i);
    img = imread(imgFile);
    imshow(img);
    title(sprintf('Obraz %d: Kliknij punkt 17 (czerwony), potem punkt 18 (żółty)', i));
    hold on;
    
    [x, y] = ginput(2);
    plot(x, y, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    text(x(1)+5, y(1), '17', 'Color', 'r', 'FontSize', 12);
    text(x(2)+5, y(2), '18', 'Color', 'y', 'FontSize', 12);

    dist_px = norm([x(2)-x(1), y(2)-y(1)]);
    cmPerPx_Imgs(i+1) = realDistImg / dist_px;

    fprintf('Obraz %d: Odległość: %.2f px | Skala: %.5f cm/px\n', ...
        i, dist_px, cmPerPx_Imgs(i+1));

    pause(0.5);
    hold off;
end

mean_cmPerPx_Img = mean(cmPerPx_Imgs);
fprintf('\nŚrednia skala z obrazów: %.5f cm/piksel\n', mean_cmPerPx_Img);

final_scale = cmPerPx / mean_cmPerPx_Img;
fprintf('Skala końcowa: %.6f\n', final_scale);
