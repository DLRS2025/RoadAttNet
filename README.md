# RoadAttNet: Remote Sensing Road Extraction with Multidimensional Features and Oriented Coordinate Attention

A Remote Sensing Image Road Extraction Algorithm Assisted by Multidimensional Features with Oriented Coordinate Attention

---

## Authors & Affiliations

**Authors:**  
Qian ZHANG<sup>1,2,3,4</sup>, Xiaomin LU<sup>1,3,4</sup>, Pengbo LI<sup>1,3,4</sup>, Zhaoyang Hou<sup>1,3,4</sup>, Qili Yang<sup>1,3,4</sup>

**Affiliations:**  
1. Faculty of Geomatics, Lanzhou Jiaotong University, Lanzhou 730070, China  
2. Lishui University, Lishui 323000, China  
3. National-Local Joint Engineering Research Center of Technologies and Applications for National Geographic State Monitoring, Lanzhou 730070, China  
4. Gansu Provincial Engineering Laboratory for National Geographic State Monitoring, Lanzhou 730070, China  

---

## Abstract

Road extraction from remote sensing imagery is challenging due to complex backgrounds, varying road widths, occlusions, and the strong linear continuity of road structures. This project presents **RoadAttNet**, an encoder–decoder segmentation framework that integrates **RGB imagery** with **fused multidimensional prior features**, and introduces an **Oriented Coordinate Attention (OCA)** mechanism to enhance road continuity and boundary delineation. The network further employs **deep supervision** with uncertainty-adaptive weighting and a composite loss combining Dice, Focal, and boundary-aware terms for robust optimization and improved extraction quality.

---

## Overview

<img width="700" height="600" alt="image" src="https://github.com/user-attachments/assets/028a4fc4-b2be-470f-8558-32b56206bc84" />

**Fig. 1.** Overall framework of this study.

---

## Model: RoadAttNet

<img width="700" height="400" alt="image" src="https://github.com/user-attachments/assets/910799a6-6c63-4f76-9e45-a816e35d14ef" />

**Fig. 2.** The architecture diagram of RoadAttNet.

---

## Datasets

### 1) Lanzhou Dataset (self-built)
The self-constructed Lanzhou Dataset is available via Baidu Netdisk (https://pan.baidu.com/s/1bWp0OVNdpVEJ1KIt5Y3oEg) 提取码: ua61.

### 2) Massachusetts Roads Dataset
- Download: https://www.cs.toronto.edu/~vmnih/data/

**Citation**
```bibtex
@phdthesis{MnihThesis,
  author = {Volodymyr Mnih},
  title  = {Machine Learning for Aerial Image Labeling},
  school = {University of Toronto},
  year   = {2013}
}
