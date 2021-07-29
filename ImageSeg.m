clc
clear variables
close all

img = imread('Vid_1.mp4 extracted.jpg');
se = strel('disk',15);
background = imopen(img,se);
imshow(background)
