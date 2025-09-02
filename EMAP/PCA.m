function dataPCA = PCA(data, PC)
%PCA 主成分分析函数，用于数据降维
%   输入：data - 原始数据矩阵，PC - 需要保留的主成分数量
%   输出：dataPCA - 降维后的数据

% 确保输入数据是二维的
if ndims(data) > 2
    error('输入数据必须是二维矩阵');
end

% 计算协方差矩阵
Sigma = cov(data);

% 计算特征值和特征向量
[V, D] = eig(Sigma);

% 提取特征值并排序（降序）
D_values = diag(D);
[D_sorted, ind] = sort(D_values, 'descend');

% 按特征值大小排序特征向量
V_sorted = V(:, ind);

% 选择前PC个主成分
V_selected = V_sorted(:, 1:PC);

% 将数据投影到选定的主成分上
dataPCA = data * V_selected;

end