function rescaled_img = rescale_img(img, min_val, max_val)
% ��ͼ�����ŵ�ָ����Χ
% ���룺img - ����ͼ��min_val - ��Сֵ��max_val - ���ֵ
% �����rescaled_img - ���ź��ͼ��

img_min = min(img(:));
img_max = max(img(:));

if img_max > img_min
    rescaled_img = min_val + (img - img_min) * (max_val - min_val) / (img_max - img_min);
else
    rescaled_img = zeros(size(img)) + min_val; % �������ֵ��ͬ����Ϊ��Сֵ
end
end