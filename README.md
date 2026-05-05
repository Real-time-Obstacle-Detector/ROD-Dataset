# ROD — Real-time Obstacle Detection Dataset

[Project: Research](https://github.com/Real-time-Obstacle-Detector) | [Format: YOLOv8](https://docs.ultralytics.com/) | [Usage: Mobile-First](https://github.com/Real-time-Obstacle-Detector/Android-Application)

**Version:** 1.0.0  
**License:** MIT

A comprehensive, high-quality YOLO-format object detection dataset specifically engineered for on-device, real-time safety. The ROD Dataset is the foundational pillar of the Real-time Obstacle Detector project, designed to empower visually impaired pedestrians and distracted walkers with instant environmental awareness.

---

## Project Vision

The Real-time Obstacle Detector (ROD) research aims to bridge the gap between complex computer vision and accessible mobile assistive technology. While many datasets focus on autonomous driving (high-speed, road-centric), ROD focuses on pedestrian-level obstacles—the small, sudden, or stationary hazards that pose daily risks on sidewalks and in urban environments.

### Research Context
This dataset was developed as part of a multi-stage research effort:
*   **[ROD Computer Vision](https://github.com/Real-time-Obstacle-Detector/Real-Time-Obstacle-Detector-ComputerVision):** A high-performance pipeline utilizing YOLOv8-nano. Through rigorous Pruning and Quantization (INT8/FP16), the model was optimized to reduce latency from seconds to milliseconds, achieving an impressive 83% mAP across all classes.
*   **[ROD Android Application](https://github.com/Real-time-Obstacle-Detector/Android-Application):** A production-ready mobile app built with Kotlin and Jetpack Compose. It leverages TensorFlow Lite and Android's NNAPI to perform real-time inference on the edge, providing haptic and audio (TTS) feedback to users without requiring an internet connection.

---

## Dataset Genesis

Building a dataset that is both diverse and accurate required a multi-faceted approach to data collection and curation:

1.  **Field Collection:** Thousands of images were captured manually using mobile devices to reflect real-world pedestrian perspectives, varied lighting conditions, and diverse urban landscapes.
2.  **Dataset Curating:** We meticulously selected and cleaned relevant subsets from massive repositories on Kaggle and Roboflow. Only images that met our strict "pedestrian-level" criteria were retained.
3.  **Merging & Refinement:** All sources were merged into a unified 25-class schema. We performed extensive label verification and noise reduction to ensure the highest quality for training low-latency models.

---

## Dataset Statistics

The current version of the dataset consists of 24,326 annotated street-level images containing over 40,000 individual object instances.

### Class Distribution

| Class Name | Total Instances | Class Name | Total Instances |
| :--- | :--- | :--- | :--- |
| **Person** | 8,962 | **Motorcycle** | 1,007 |
| **Car** | 5,578 | **Traffic sign** | 907 |
| **Manhole** | 2,501 | **Traffic Cone** | 941 |
| **Pedestrian crosswalk** | 2,493 | **Truck** | 823 |
| **Chair** | 2,344 | **Bench** | 774 |
| **Tree** | 2,075 | **Dog** | 728 |
| **Stairs** | 1,687 | **Road** | 682 |
| **Guard rail** | 1,585 | **Plant Pot** | 631 |
| **Dustbin** | 1,549 | **Electrical Pole** | 618 |
| **Bus** | 1,214 | **Traffic Barrel** | 576 |
| **Building** | 1,041 | **Electrical Box** | 513 |
| **Bike** | 467 | **Fire hydrant** | 255 |
| **Bicycle Rack** | 244 | | |

**Total Instances Across All Splits:** 40,195

---

## Key Features

*   **Mobile Optimized:** Specifically designed for training small-footprint models (YOLOv8n, MobileNet, etc.) that run smoothly on mobile CPUs/NPs.
*   **Urban Centric:** Focuses on objects like Manholes, Stairs, and Electrical Poles that are often ignored in standard COCO or KITTI datasets.
*   **Ready-to-Use:** Provided in standard YOLO format with pre-defined train, valid, and test splits for immediate experimentation.

---

## License

This dataset and its associated scripts are released under the **MIT License**.

---

## Citation

If you use this dataset or the ROD models in your work, please cite the original research repositories.

> Research by the Real-time Obstacle Detector Team.
