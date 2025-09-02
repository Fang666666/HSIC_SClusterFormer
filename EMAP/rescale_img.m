function rescaled_img = rescale_img(img, min_val, max_val)
% 将图像缩放到指定范围
% 输入：img - 输入图像，min_val - 最小值，max_val - 最大值
% 输出：rescaled_img - 缩放后的图像

img_min = min(img(:));
img_max = max(img(:));

if img_max > img_min
    rescaled_img = min_val + (img - img_min) * (max_val - min_val) / (img_max - img_min);
else
    rescaled_img = zeros(size(img)) + min_val; % 如果所有值相同，设为最小值
end
end