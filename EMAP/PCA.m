function dataPCA = PCA(data, PC)
%PCA ���ɷַ����������������ݽ�ά
%   ���룺data - ԭʼ���ݾ���PC - ��Ҫ���������ɷ�����
%   �����dataPCA - ��ά�������

% ȷ�����������Ƕ�ά��
if ndims(data) > 2
    error('�������ݱ����Ƕ�ά����');
end

% ����Э�������
Sigma = cov(data);

% ��������ֵ����������
[V, D] = eig(Sigma);

% ��ȡ����ֵ�����򣨽���
D_values = diag(D);
[D_sorted, ind] = sort(D_values, 'descend');

% ������ֵ��С������������
V_sorted = V(:, ind);

% ѡ��ǰPC�����ɷ�
V_selected = V_sorted(:, 1:PC);

% ������ͶӰ��ѡ�������ɷ���
dataPCA = data * V_selected;

end