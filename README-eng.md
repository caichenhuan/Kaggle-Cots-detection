# Kaggle-Cots-detection

Using object detection to detect starfish in underwater videos.

Competition link: [Kaggle: TensorFlow - Help Protect the Great Barrier Reef](https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/overview)

## Introduction

### Background

Nowadays, the reef is under threat, in part because of the overpopulation of one particular starfish – the coral-eating crown-of-thorns starfish (or COTS for short). Scientists, tourism operators, and reef managers established a large-scale intervention program to control COTS outbreaks to ecologically sustainable levels.

### Tasks

The goal of this competition is to accurately identify starfish in real-time by building an object detection model trained on underwater videos of coral reefs.

### Evaluation

This competition is evaluated on the $F_2-Score$ at different intersections over union (IoU) thresholds. The metric sweeps over IoU thresholds in the range of 0.3 to 0.8 with a step size of 0.05, calculating an $F_2-Score$ at each threshold. The final $F_2-Score$ is calculated as the mean of the F2 scores at each IoU threshold.
$$
F_\beta-Score=(1+{\beta}^2)\cdot \frac{Percision \cdot Recall}{\beta^2 \cdot Percision \cdot Recall}\\Precision=\frac{TP}{TP+FP}~~~~Recall=\frac{TP}{TP+FN}
$$

## Dataset

The source of the dataset: [The CSIRO Crown-of-Thorn Starfish Detection Dataset](https://arxiv.org/abs/2111.14311)



![sample_pic_1](G:\project\Kaggle-Cots-Detection\assets\readme\sample_pic_1.jpg)





![sample_video_1](G:\project\Kaggle-Cots-Detection\assets\readme\sample_video_1.gif)



























