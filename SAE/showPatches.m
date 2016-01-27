load patches

for i=1:1:1000
    img = patches(:,i);
    img2 = reshape(img, 8,8);
    imagesc(img2), colormap gray;
end 