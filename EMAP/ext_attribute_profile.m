function EAP = ext_attribute_profile(I, attr, lambdas)

% Rescale the images
I = uint16(rescale_img(I, 0, 1000));
EAP = [];

for i=1:size(I,3)
    eap_i = attribute_profile(I(:,:,i), attr, lambdas);
    EAP = cat(3, EAP, eap_i);
end