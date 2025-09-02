% FE_EMAP.m - �������ļ�
% ������ȡ��չ��̬������������

% ����������������
clear;
clc;

% �������ݣ�������Ҫ�滻Ϊ����ʵ�����ݼ��ش��룩
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

% ��������Ƿ��Ѽ���
if ~exist('data', 'var') || ~exist('data_gt', 'var')
    error('���ȼ������ݵ���������data �� data_gt ����');
end

% ��ʼ��
[rows, cols] = size(data_gt);

% ȷ�������Ƕ�ά��
if ndims(data) > 2
    % ��������Ƕ�ά�ģ���������Ϊ��ά
    data = reshape(data, size(data, 1)*size(data, 2), size(data, 3));
    disp('�����ѴӶ�ά����Ϊ��ά��ʽ');
end

% ���� PCA ά��
d_pca = 7;

% ִ�� PCA ��ά
Dp = PCA(data, d_pca);

% �� PCA ������ܻ�ԭʼ�ռ�ά��
Dp_reshaped = reshape(Dp, rows, cols, d_pca);

%% �ռ�������ȡ

%% ���� EAP����չ�������棩

%% EAP �������
attr = 'area';
lambdas = [100, 500, 1000, 5000];
EAP = ext_attribute_profile(reshape(Dp, rows, cols, d_pca), attr, lambdas);
EAP_a = double(reshape(EAP, rows*cols, size(EAP, 3)));

%% EAP ��������
attr = 'inertia';
lambdas = [0.2, 0.3, 0.4, 0.5];
EAP = ext_attribute_profile(reshape(Dp, rows, cols, d_pca), attr, lambdas);
EAP_i = double(reshape(EAP, rows*cols, size(EAP, 3)));

%% EAP ��׼������
attr = 'std';
lambdas = [20, 30, 40, 50];
EAP = ext_attribute_profile(reshape(Dp, rows, cols, d_pca), attr, lambdas);
EAP_s = double(reshape(EAP, rows*cols, size(EAP, 3)));

%% �ϲ����������γ� EMAP
EMAP = [EAP_a, EAP_i, EAP_s];

% �� EMAP ���� PCA ��ά
npca = 30;
EMAP = PCA(EMAP, 1);
Feature_E = reshape(EMAP, rows, cols, 1);

% ȷ������Ŀ¼����
if ~exist('EMAP_Data', 'dir')
    mkdir('EMAP_Data');
end

% ������
save('EMAP_Data/HongHu_EMAP.mat', 'Feature_E');
disp('������ȡ��ɣ�����ѱ��浽 EMAP_Data/HongHu_EMAP.mat');

% ��ʾ�����Ϣ
disp('FE_EMAP ������ɣ�');