# [CVPR 2023] Bridging Precision and Confidence: A Train-Time Loss for Calibrating Object Detection

[Paper Link](arxivlink)

This paper is accepted at CVPR 2023 and this repositoy contains the PyTorch implementation.

## Abstract
Deep neural networks (DNNs) have enabled astounding progress in several vision-based problems. Despite showing high predictive accuracy, recently, several works have revealed that they tend to provide overconfident predictions and thus are poorly calibrated. The majority of the works addressing the miscalibration of DNNs fall under the scope of classification and consider only in-domain predictions. However, there is little to no progress in studying the calibration of DNN-based object detection models, which are central to many vision-based safety-critical applications. In this paper, inspired by the train-time calibration methods, we propose a novel auxiliary loss formulation that explicitly aims to align the class confidence of bounding boxes with the accurateness of predictions (i.e. precision). Since the original formulation of our loss depends on the counts of true positives and false positives in a minibatch, we develop a differentiable proxy of our loss that can be used during training with other application-specific loss functions.

![alt text](RD_COCO_img.png)

Reliability Diagrams: Selected classes from MS-COCO (In-Domain) and CorCOCO (Out-Domain). Top: Baseline trained as D-DETR and Bottom: D-DETR trained with our proposed BPC loss.

## Setup
