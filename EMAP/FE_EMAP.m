% FE_EMAP.m - 主程序文件
% 用于提取扩展形态属性剖面特征

% 清除工作区和命令窗口
clear;
clc;

% 加载数据（这里需要替换为您的实际数据加载代码）
% load('../data/PaviaU.mat');
% load('../data/PaviaU_gt.mat');
% data = paviaU;
% data_gt = paviaU_gt;

% load('../data/WHU_Hi_HanChuan.mat');
% load('../data/WHU_Hi_HanChuan_gt.mat');
% data = WHU_Hi_HanChuan;
% data_gt = WHU_Hi_HanChuan_gt;

load('../data/WHU_Hi_HongHu.mat');
load('../data/WHU_Hi_HongHu_gt.mat');
data = WHU_Hi_HongHu;
data_gt = WHU_Hi_HongHu_gt;

% 检查数据是否已加载
if ~exist('data', 'var') || ~exist('data_gt', 'var')
    error('请先加载数据到工作区：data 和 data_gt 变量');
end

% 初始化
[rows, cols] = size(data_gt);

% 确保数据是二维的
if ndims(data) > 2
    % 如果数据是多维的，将其重塑为二维
    data = reshape(data, size(data, 1)*size(data, 2), size(data, 3));
    disp('数据已从多维重塑为二维格式');
end

% 设置 PCA 维度
d_pca = 7;

% 执行 PCA 降维
Dp = PCA(data, d_pca);

% 将 PCA 结果重塑回原始空间维度
Dp_reshaped = reshape(Dp, rows, cols, d_pca);

%% 空间特征提取

%% 计算 EAP（扩展属性剖面）

%% EAP 面积属性
attr = 'area';
lambdas = [100, 500, 1000, 5000];
EAP = ext_attribute_profile(reshape(Dp, rows, cols, d_pca), attr, lambdas);
EAP_a = double(reshape(EAP, rows*cols, size(EAP, 3)));

%% EAP 惯性属性
attr = 'inertia';
lambdas = [0.2, 0.3, 0.4, 0.5];
EAP = ext_attribute_profile(reshape(Dp, rows, cols, d_pca), attr, lambdas);
EAP_i = double(reshape(EAP, rows*cols, size(EAP, 3)));

%% EAP 标准差属性
attr = 'std';
lambdas = [20, 30, 40, 50];
EAP = ext_attribute_profile(reshape(Dp, rows, cols, d_pca), attr, lambdas);
EAP_s = double(reshape(EAP, rows*cols, size(EAP, 3)));

%% 合并所有属性形成 EMAP
EMAP = [EAP_a, EAP_i, EAP_s];

% 对 EMAP 进行 PCA 降维
npca = 30;
EMAP = PCA(EMAP, 1);
Feature_E = reshape(EMAP, rows, cols, 1);

% 确保保存目录存在
if ~exist('EMAP_Data', 'dir')
    mkdir('EMAP_Data');
end

% 保存结果
save('EMAP_Data/HongHu_EMAP.mat', 'Feature_E');
disp('特征提取完成，结果已保存到 EMAP_Data/HongHu_EMAP.mat');

% 显示完成信息
disp('FE_EMAP 处理完成！');