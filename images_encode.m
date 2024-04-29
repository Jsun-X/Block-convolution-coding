% 载入图像数
imageDir = fullfile('images');
imds = imageDatastore(imageDir, 'IncludeSubfolders', true, 'FileExtensions', '.jpg');

% 调整图像大小和格式化
imds.ReadFcn = @(filename)imresize(imread(filename), [64, 64]);

% 预分配数组（假设图像为灰度图，如果是彩色，第三维应该是3）
numImages = numel(imds.Files);
images = zeros(64, 64, 1, numImages);  % 修改为3如果是RGB图像

% 读取图像并存储到数组
for idx = 1:numImages
    img = readimage(imds, idx);
    if size(img, 3) == 1
        images(:, :, 1, idx) = img;
    else
        images(:, :, :, idx) = rgb2gray(img);  % 如果原始是彩色，转为灰度
    end
end

% 确保数据类型为 double（如果需要）
images = double(images);

% 定义网络层
layers = [
    imageInputLayer([64 64 1], 'Name', 'input')
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
    
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
    
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv3')
    reluLayer('Name', 'relu3')
    transposedConv2dLayer(2, 16, 'Stride', 2, 'Cropping', 'same', 'Name', 'transConv1')
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv4')
    reluLayer('Name', 'relu4')
    transposedConv2dLayer(2, 32, 'Stride', 2, 'Cropping', 'same', 'Name', 'transConv2')
    
    convolution2dLayer(3, 1, 'Padding', 'same', 'Name', 'conv5')
    reluLayer('Name', 'relu5')
    convolution2dLayer(1, 1, 'Padding', 'same', 'Name', 'output')
];

% 连接输出层
layers = [layers; regressionLayer('Name', 'regression')];

% 定义训练选项
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 3000, ...
    'MiniBatchSize', 128, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress');

% 训练自动编码器
net = trainNetwork(images, images, layers, options);

% 假设您已经训练了自动编码器网络 net，并且有训练数据 images

% 从训练数据中选择一些图像作为测试图像
numTestImages = 5;  % 指定测试图像的数量
testIndices = randperm(size(images, 4), numTestImages);  % 随机选择测试图像的索引
testImages = images(:, :, :, testIndices);  % 从训练数据中获取测试图像

% 选择一个测试图像作为输入
sampleIdx = 1;
inputImage = testImages(:, :, :, sampleIdx);

% 使用训练好的网络进行图像重建
reconstructedImage = predict(net, inputImage);

% 显示原始图像和重建图像
figure;
subplot(1, 2, 1);
imshow(inputImage);
title('Original Image');

subplot(1, 2, 2);
imshow(reconstructedImage);
title('Reconstructed Image');

% 计算重建图像与原始图像之间的差异（如果需要）
diffImage = abs(inputImage - reconstructedImage);
mse = mean(diffImage(:).^2);
fprintf('Mean Squared Error (MSE) between original and reconstructed image: %f\n', mse);


% 检查 images 文件夹路径是否正确
imageDir = 'images';  % images 文件夹路径

% 读取图像
imds = imageDatastore(imageDir, 'IncludeSubfolders', true, 'FileExtensions', {'.jpg', '.png', '.jpeg'});

% 调整图像大小和格式化
imds.ReadFcn = @(filename)imresize(imread(filename), [64, 64]);

% 读取图像数据
images = readall(imds);

% 将图像数据转换为数值数组
num_images = numel(images);
for i = 1:num_images
    images{i} = double(images{i});
end

% 将 cell 数组转换为数值数组
images = cat(4, images{:});

% 现在，您可以使用这个数值数组进行编码和解码操作
% 将 RGB 彩色图像转换为灰度图像
gray_images = zeros(64, 64, 1, size(images, 4));
for i = 1:size(images, 4)
    % 从 RGB 彩色图像中提取每个通道
    red_channel = images(:, :, 1, i);
    green_channel = images(:, :, 2, i);
    blue_channel = images(:, :, 3, i);
    
    % 使用加权平均值将 RGB 通道转换为灰度值
    gray_image = 0.2989 * red_channel + 0.5870 * green_channel + 0.1140 * blue_channel;
    
    % 存储灰度图像
    gray_images(:, :, 1, i) = gray_image;
end

% 使用训练好的网络进行编码和解码，并恢复图像
reconstructed_images = predict(net, gray_images);

% 显示原始图像和重建图像

num_images = size(images, 4); % 假设 images 是四维数组
figure; % 创建一个图形窗口

for i = 1:num_images
    % 原始图像
    original_image = images(:, :, :, i);
    % 重建图像
    reconstructed_image = reconstructed_images(:, :, :, i);
    
    % 显示原始图像
    subplot(num_images, 2, 2*i-1); % 总行数为num_images，总列数为2，当前位置为奇数列（1, 3, 5, ...）
    imshow(uint8(original_image));
    title(sprintf('Original Image %d', i));
    
    % 显示重建图像
    subplot(num_images, 2, 2*i); % 总行数为num_images，总列数为2，当前位置为偶数列（2, 4, 6, ...）
    imshow(uint8(reconstructed_image));
    title(sprintf('Reconstructed Image %d', i));
end




