# Video Labeling Process Description

## Overview

This document describes the manual video labeling process used to create ground truth annotations for the activity recognition dataset. The labeling process was conducted on video recordings from 16 subjects to establish activity labels for training and evaluation of machine learning models.

## Data Coverage

Approximately **70% of the available data** from the 16 subjects was manually labeled through video observation by me. This selective labeling approach focused on capturing representative samples of various activities while maintaining a manageable annotation workload. The remaining labels originete from the labeling process of another student.

## Annotation Software

The video labeling was performed using the VIA (VGG Image Annotator) annotation software, which provided efficient tools for temporal annotation of video content.

**Reference:**
```
@inproceedings{dutta2019vgg,
  author = {Dutta, Abhishek and Zisserman, Andrew},
  title = {The {VIA} Annotation Software for Images, Audio and Video},
  booktitle = {Proceedings of the 27th ACM International Conference on Multimedia},
  series = {MM '19},
  year = {2019},
  isbn = {978-1-4503-6889-6/19/10},
  location = {Nice, France},
  numpages = {4},
  url = {https://doi.org/10.1145/3343031.3350535},
  doi = {10.1145/3343031.3350535},
  publisher = {ACM},
  address = {New York, NY, USA},
}
```

## Temporal Synchronization Challenges

The labeling process faced significant temporal alignment challenges:

- **Camera clock drift**: Discrepancies between camera timestamps and sensor data timestamps
- **Human synchronization errors**: Manual errors introduced during the synchronization events
- **Time shifts**: Considerable temporal offsets between video annotations and corresponding sensor measurements

These synchronization issues required subsequent correction procedures to align the video-based labels with the sensor data timeline.

## Labeling Output

The manual video annotation process resulted in approximately **145 unique activity labels** across all subjects. These labels serve as ground truth for training and evaluating activity recognition models on the sensor data.

## Data Processing Pipeline

The labeled data underwent further processing steps including:
- Temporal alignment correction
- Label validation and cleaning
- Integration with sensor data streams
- Feature extraction and windowing for machine learning applications