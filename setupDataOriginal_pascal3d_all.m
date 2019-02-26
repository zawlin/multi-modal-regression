function setupDataOriginal_pascal3d_all
% function to setup pascal3d+ data with flips. Read the data from PASCAL3D+_release1.1
% stored in db_path. Save resized patch inside ground truth bounding box.
% Usage: setupDataOriginal_pascal3d(cls, db_path, voc_dir);
% cls: class of interest
% db_path: location of the pascal3d+ data. eg: 'D:/datasets/pascal3d/PASCAL3D+_release1.1/';
% voc_dir: location of VOC2012 devkit to get train+val sets. eg: 'D:/datasets/VOCdevkit/VOC2012';

clear; clc;

classes = {'aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', ...
	'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor'};
% start parallel processing
%poolobj = parpool(16);


for k=1:length(classes)
	cls = classes{k};
    setupDataOriginal_pascal3d(cls);
end

%delete(poolobj);
