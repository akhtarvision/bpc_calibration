# [CVPR 2023] Bridging Precision and Confidence: A Train-Time Loss for Calibrating Object Detection

[Paper Link](arxivlink)

Authors: [Muhammad Akhtar Munir](https://scholar.google.com.pk/citations?user=sT-epZAAAAAJ&hl=en), [Muhammad Haris Khan](https://scholar.google.com.pk/citations?user=ZgERfFwAAAAJ&hl=en), [Salman Khan](https://salman-h-khan.github.io/), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en)

This paper is accepted at CVPR 2023 and this repositoy contains the PyTorch implementation.

## Abstract
Deep neural networks (DNNs) have enabled astounding progress in several vision-based problems. Despite showing high predictive accuracy, recently, several works have revealed that they tend to provide overconfident predictions and thus are poorly calibrated. The majority of the works addressing the miscalibration of DNNs fall under the scope of classification and consider only in-domain predictions. However, there is little to no progress in studying the calibration of DNN-based object detection models, which are central to many vision-based safety-critical applications. In this paper, inspired by the train-time calibration methods, we propose a novel auxiliary loss formulation that explicitly aims to align the class confidence of bounding boxes with the accurateness of predictions (i.e. precision). Since the original formulation of our loss depends on the counts of true positives and false positives in a minibatch, we develop a differentiable proxy of our loss that can be used during training with other application-specific loss functions.

![alt text](RD_COCO_img.png)

Reliability Diagrams: Selected classes from MS-COCO (In-Domain) and CorCOCO (Out-Domain). Top: Baseline trained as D-DETR and Bottom: D-DETR trained with our proposed BPC loss.

## Setup

Requirements: This implementation is on:

Linux, CUDA>=11.0

Python>=3.7

PyTorch>=1.7.0

## Installation, Training and Evaluation

For complete Installation, Training and Evaluation commands, follow guidelines [here](https://github.com/fundamentalvision/Deformable-DETR#installation)



## Citation

Please cite the following, if you find this work useful in your research:
```bibtex
@InProceedings{munirs_bpc_CVPR23,
    author    = {Munir, Muhammad Akhtar and Khan, Muhammad Haris and Khan, Salman and Khan, Fahad Shahbaz},
    title     = {Bridging Precision and Confidence: A Train-Time Loss for Calibrating Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {},
    year      = {2023},
    pages     = {}
}
```

## Contact
In case of any query, create issue or contact akhtar.munir@mbzuai.ac.ae 

## Acknowledgement
This codebase is built on <a href="https://github.com/fundamentalvision/Deformable-DETR">Deformable-DETR</a> and <a href="https://github.com/rafaelpadilla/Object-Detection-Metrics">Object Detection Metrics</a>
