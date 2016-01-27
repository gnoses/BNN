load IMAGES;

for i=1:1:10
    colormap('gray');
    image(IMAGES(:,:,i).*255);
end