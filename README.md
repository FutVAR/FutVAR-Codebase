[![Release Notes](https://img.shields.io/badge/releases-view-blue)](https://github.com/FutVAR)
[![Contribution](https://img.shields.io/badge/contribute-welcome-green)](https://github.com/FutVAR)
[![Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-view-white)](https://github.com/FutVAR)
[![License](https://img.shields.io/badge/license-n/a-red)](https://github.com/FutVAR)
[![Versions](https://img.shields.io/badge/versions-1.0.0-orange)](https://github.com/FutVAR)
[![Documentation](https://img.shields.io/badge/documentation-view-violet)](https://github.com/FutVAR)
[![Discussions](https://img.shields.io/badge/discussions-view-yellow)](https://github.com/FutVAR)
# FutVAR (Futebol Video Analytics and Reporting System)
![FutVAR](img/futvar-bg.jpg)
Football, a sport where every second has its significance, a sport where every centimeter and every second matter. That's why we decided to do a project on Football as a testing ground to push our object detection, image segmentation, keypoint detection, and foundational models to their limits. In the sports industry, precision and quick decision-making play a crucial role in determining success.

## Table of Contents

- [Introduction](##introduction)
- [Features](#f#eatures)
- [Requirements](##requirements)
- [Models](##models)
- [Challenges](##challenges)
- [Conclusion](##conclusion)

## Introduction
FutVAR stands for "Futebol Video Analytics and Reporting System". FutVAR is a football video analytics and monitoring system that leverages computer vision technology to revolutionize traditional methodologies in  football analysis. By utilizing advanced object detection models like YOLO, ByteTrack, Faster R-CNN, and Detectron2, and leveraging OpenCV for achieving state-of-the-art performance on image and video processing, FutVAR provides a comprehensive analysis of key football metrics, including passes, fouls, possessions, goals, and other critical game evaluations. FutVAR ensures the integrity of rules and regulations by accurately tracking players, the ball, and the ball's path; distinguishing between teams by their jerseys; identifying referees and goalkeepers; and detecting red and yellow card incidents. The system employs sophisticated video processing techniques using OpenCV to extract meaningful insights from game footage, enhancing the quality and speed of analysis. By automating these game analysis aspects, FutVAR offers a reliable tool for enhancing the development and evaluation of football matches. With capabilities like real-time video processing, motion tracking, and event detection, FutVAR is a game-changer in football analytics.

## Requirements
The following requirements are used in this project and to run this project, you need to have installed these modules:
- Python 3.x
- ultralytics
- autodistil
- detectron2
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas, etc.

## Datasets
| Detection                       | Dataset                                                                                                                                                          |
|:--------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Ball detection         | [![Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/futvar-n32uw/futvar/) |
| Player detection         | [![Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/futvar-n32uw/futvar/) |
| Goal Keeper detection           | [![Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/futvar-n32uw/futvar/)    |
| Referee detection | [![Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/futvar-n32uw/futvar/)   |

## Models
- Grounding DINO
- YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLOv11
- Faster R-CNN
- DETR
- CNN-RNN
- CNN-LSTM
- Transformer

## Challenges
We welcome contributions from anyone who loves computer vision and shares our passion! Together, we can build powerful open-source tools for sports analytics. Here are the main challenges we're looking to tackle:

1. Ball tracking: Tracking the ball is extremely difficult due to its small size and rapid movements, especially in high-resolution videos.
2. Reading jersey numbers: Accurately reading player jersey numbers is often hampered by blurry videos, players turning away, or other objects obscuring the numbers.
3. Player tracking: Maintaining consistent player identification throughout a game is a challenge due to frequent occlusions caused by other players or objects on the field.
4. Player re-identification: Re-identifying players who have left and re-entered the frame is tricky, especially with moving cameras or when players are visually similar.
5. Camera calibration: Accurately calibrating camera views is crucial for extracting advanced statistics like player speed and distance traveled. This is a complex task due to the dynamic nature of sports and varying camera angles.

## Conclusion
This project aims to detect and track players, referees, and footballs in a video using YOLO, one of the best AI object detection models available. We will also train the model to improve its performance. Additionally, we will assign players to teams based on the colors of their t-shirts using Kmeans for pixel segmentation and clustering. With this information, we can measure a team's ball acquisition percentage in a match. We will also use optical flow to measure camera movement between frames, enabling us to measure a player's movement accurately. Furthermore, we will implement perspective transformation to represent the scene's depth and perspective, allowing us to measure a player's movement in meters rather than pixels. Finally, we will calculate a player's speed and the distance covered. This project covers various concepts and addresses real-world problems, making it suitable for both beginners and experienced machine learning engineers.

<!--
## contribution

We love your input! Please see our [contributing guide](https://github.com/futvar/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!

<p align="center">
    <a href="https://github.com/roboflow/supervision/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=roboflow/supervision" />
    </a>
</p>
-->
