# Motion-Guided Small MAV Detection in Complex and Non-Planar Scenes

> **arXiv: 2410.10527v1 [cs.CV], 14 Oct 2024** · License: CC BY-NC-SA 4.0
>
> **Hanqing Guo**, **Canlun Zheng**, **Shiyu Zhao** (corresponding author)
>
> H. Guo is with the College of Computer Science and Technology, Zhejiang University and also with the School of Engineering, Westlake University, Hangzhou, China (e-mail: guohanqing@westlake.edu.cn). C. Zheng and S. Zhao are with the School of Engineering at Westlake University, Hangzhou, China (e-mail: zhaoshiyu@westlake.edu.cn).

## Abstract

In recent years, there has been a growing interest in the visual detection of micro aerial vehicles (MAVs) due to its importance in numerous applications. However, the existing methods based on either appearance or motion features encounter difficulties when the background is complex or the MAV is too small. In this paper, we propose a novel motion-guided MAV detector that can accurately identify small MAVs in complex and non-planar scenes. This detector first exploits a motion feature enhancement module to capture the motion features of small MAVs. Then it uses multi-object tracking and trajectory filtering to eliminate false positives caused by motion parallax. Finally, an appearance-based classifier and an appearance-based detector that operates on the cropped regions are used to achieve precise detection results. Our proposed method can effectively and efficiently detect extremely small MAVs from dynamic and complex backgrounds because it aggregates pixel-level motion features and eliminates false positives based on the motion and appearance features of MAVs. Experiments on the ARD-MAV dataset demonstrate that the proposed method could achieve high performance in small MAV detection under challenging conditions and outperform other state-of-the-art methods across various metrics.

**Index Terms:** Motion detection, MAV detection, Multi-object tracking.

---

## I. Introduction

Vision-based air-to-air MAV detection has attracted increasing attention in recent years due to its application in many tasks such as vision-based swarming [1], aerial see-and-avoid [2], and malicious MAV detection [3]. Air-to-air MAV detection is *more challenging* than general object detection because the camera is moving and the target MAV is often engulfed by complex background scenes such as buildings and trees when the camera looks down from the top. Moreover, the target MAV may be extremely small in the image when seen from a distance.

![Fig.1(a) Complex background](assets/fig1_a.png)

![Fig.1(b) Small MAV targets (8x8 pixels)](assets/fig1_b.png)

**Fig. 1:** Examples of challenging conditions for MAV detection. In air-to-air MAV detection scenarios, the target MAV is often engulfed by a complex background or may be extremely small in the image. The yellow boxes enclose MAVs. (Top right images are a better view with 300% zoom-in).

Recently, many appearance-based methods that rely on deep learning techniques have been proposed for vision-based MAV detection. For example, some state-of-the-art object detection networks such as YOLO series, R-CNN series, SSD, and DETR have been applied to MAV detection [4], [5], [6]. These methods can work effectively in relatively simple scenarios where the target MAV is distinct from the background and its size is relatively large in the image. However, they often fail in more complex scenarios since appearance features are *not reliable*. For example, as shown in Fig. 1(a), when the background scene is extremely complex, the target MAV can easily get engulfed in the background scenes. Moreover, when the target MAV flies far away from the camera, its image may only occupy a tiny portion of the image. For example, as shown in Fig. 1(b), an MAV seen from about 100 m away only occupies 8×8 pixels in an image of 1920×1080 pixels. Therefore, high-performance algorithms must be developed to detect MAVs under such challenging conditions.

Motion features can greatly help detect MAVs under challenging conditions. Many motion-assisted methods introduce motion features into MAV detection based on, for example, background subtraction [2], [7], low-rank based methods [8], [9], spatio-temporal information [10], [11], and optical flow [12], [13]. However, motion-assisted MAV detection still faces the following *challenges*. First, motion cues of MAVs are difficult to separate from the background when the target size is too small or the background is too complex. For example, we tested the popular optical flow model RAFT [14] for motion feature extraction. The result is given in Fig. 2, which shows that the dense optical flow network can successfully extract the motion information when the target has a high contrast to the background or is large enough. However, when the background is complex or the target is small, the motion information can hardly be extracted. Second, the influence of image alignment error or parallax effects is significant when the camera is moving. Most of the existing motion-assisted methods only deal with *planar* backgrounds [2], [7], [15] so that affine transformation or 2D perspective transformation can be used to align adjacent frames. Although the planar assumption is valid when the camera flies at a high altitude so that the height of the ground objects is negligible, it is still invalid in many cases when the flight altitude is relatively low. Objects on the ground such as buildings, trees, and lampposts may violate the planar assumption and result in many false positives.

![Fig.2(a) RAFT succeeds](assets/fig2_1.png)

![Fig.2(b) RAFT fails](assets/fig2_2.png)

**Fig. 2:** Testing results of the dense optical flow of moving MAVs extracted by the pre-trained model of the RAFT [14]. **Yellow box** indicates the MAV target. **Black box** indicates the extracted dense optical flow.

To overcome the limitations of the existing approaches, we propose a motion-guided MAV detector (MGMD) for small MAV detection in complex and non-planar scenes. This algorithm consists of three modules: motion feature enhancement, trajectory filtering, and local fine detection. First, a motion feature enhancement module is applied to extract the motion features of small MAVs from the complex background. Second, multi-object tracking is used to generate trajectories for each target and a trajectory-based classifier is utilized to eliminate false positives. Third, an appearance-based classifier and an appearance-based detector that operates on the cropped regions are used to achieve precise detection results.

The technical novelties are summarized as follows.

- Compared with traditional background subtraction and optical flow, our proposed motion feature enhancement module can effectively extract motion features of extremely small MAVs from dynamic and complex backgrounds. This is because we aggregate pixel-level motion features by frame alignment and multi-frame differences. Our proposed detector outperforms the state-of-the-art algorithms such as [12], [16], [17] on various evaluation metrics.
- The influence of false positives generated by motion parallax can be effectively removed by multi-object tracking, trajectory filtering, and appearance-based classification. This is because our proposed method can model the spatial and temporal features of MAVs that are different from the false positives generated by motion parallax. The effectiveness of this method is validated by experimental results on the ARD-MAV dataset where 3D structures such as tall buildings, trees, and lampposts are dominant.

## II. Related work

### II-A. Vision-based MAV detection

Vision-based MAV detection has been a hot topic in recent years. Due to the great success on general object detection, some researchers have directly applied the state-of-the-art object detection networks for MAV detection. The work in [5] evaluates eight state-of-the-art deep learning algorithms on the Det-Fly dataset for MAV detection. Similarly, the authors in [4] evaluate four state-of-the-art deep learning algorithms on three representative MAV datasets. To further improve the detection accuracy, the authors in [18] propose a novel approach that combines transfer learning and adaptive fusion mechanism to improve small object detection performance. However, the experimental results in these works show that the existing object detection networks do not perform well in challenging conditions such as complex backgrounds, motion blur, and small objects due to insufficient visual features.

To improve the detection accuracy under challenging conditions, the motion feature is introduced to assist MAV detection. [19] is an early work in the field of detecting flying objects using a moving camera. The authors first employ two CNN networks in a sliding window fashion to obtain the motion-stabilized spatial-temporal cubes and then use a third CNN network to classify MAV in each spatial-temporal cube. Similarly, the work in [7] proposes a UAV-to-UAV video dataset and a background-subtraction-based method for small UAV detection. Subsequently, [12] proposes a two-stage segmentation-based approach for detecting drones from drone videos and [13] proposes a feature super-resolution-based UAV detector with motion information extracted by dense optical flow. Recently, the work in [20] proposes an end-to-end method for drone-to-drone detection. The authors utilize CSPDarkNet53 to learn spatial features and VideoSwin Transformer to exploit temporal information. These methods could work fine when targets occupy a relatively large part of the entire image or the background is comparatively simple. However, when handling extremely small targets (such as 10×10 pixels) in a cluttered environment, the detection accuracy degrades vastly.

### II-B. Small object detection in aerial images

Object detection in aerial images has been a popular task in recent years. Due to the extremely small object sizes and numerous background clutters, the general object detection algorithms usually perform poorly in such scenarios. [16] tries to improve the object detection accuracy on drone-captured scenarios by adding the Transformer Prediction Heads and CBAM to YOLOv5. The work in [21] proposes a dense-and-similar object detector with four key components: a coarse detector, an adaptive clustering procedure, a similar-class classifier, and a fine detector. Recently, the work in [22] aggregates abundant object-oriented information in low-resolution imagery to enhance the recognition ability of small objects in large-scale images. In addition, temporal context has also been introduced to enhance the feature of small targets. The work in [23] proposes a novel two-stage spatio-temporal CNN which effectively and efficiently combines both appearance and motion information. The work in [24] presents a spatiotemporal deep learning model based on YOLOv5 that exploits temporal context by processing sequences of frames at once. Although these methods are shown to be effective in certain small object detection scenarios, their effectiveness for small MAV detection has not been verified yet. Moreover, the computational cost of them is usually unacceptable for real-world applications.

![Fig.3](assets/fig3.png)

**Fig. 3:** The architecture of the proposed MGMD algorithm. First, we use frame alignment and three-frame difference to segment moving objects from complex backgrounds. Then, multi-object tracking is applied to generate trajectories for each detection, and a trajectory-based filter is used to eliminate false positives. Finally, a local appearance-based classifier and a local appearance-based detector are utilized on a cropped region to obtain precise detection results.

## III. Proposed method

The architecture of our proposed algorithm is shown in Fig. 3. It is composed of three modules: motion feature enhancement (MFE), trajectory filtering (TF), and local fine detection. The following sections detail each module.

### III-A. Motion feature enhancement

Since the MAV is difficult to detect when its size is too small or the background is too complex, we first design a motion feature enhancement module to obtain the potential areas that probably contain moving MAVs.

Motion feature extraction or moving object detection is a well-studied problem in constrained environments where the camera's ego motion is not obvious. However, when it comes to unconstrained ego motion, moving object detection becomes a much harder problem. Inspired by related works [7], [25], [26], we first exploit image alignment techniques to obtain stabilized frames, and then frame difference and morphological operations are performed to segment moving objects from multiple frames. Our motion feature enhancement module can be divided into three steps: frame alignment, motion feature extraction, and post-processing.

#### III-A1. Frame alignment

To separate moving objects from the dynamic background, we must first align successive frames so that the influence of the camera's ego motion can be eliminated. In this paper, 2D perspective transformation is used for motion compensation because it exactly models the 2D background motion when the background results from the relative motion of a 2D plane in the 3D world. In principle, the perspective transformation requires the background to be planar or the camera only rotates. In some cases, this is a reasonable approximation since the MAV looks at the ground from a high altitude. However, when the flight altitude is relatively low, 3D objects such as tall buildings, lampposts, trees, and wire poles can remarkably influence the motion compensation quality and generate many false positives. Nevertheless, our proposed trajectory-based filter and the appearance-based classifier are applied to remove these false positives.

For computational efficiency and robustness in textureless regions such as sky and grassland, grid-based key points are used instead of the feature detectors to calculate the homography matrix. We sample $30 \times 20$ key points uniformly distributed in each row and column across the previous $k$ frame. Then, these key points are tracked by the pyramidal Lucas-Kanade (LK) algorithm to obtain the corresponding points in the current frame. After these key points are matched over two successive frames, the homography matrix $H$ is calculated with the RANSAC method to reject outliers. The image in the previous $k$ frame $I_{t-k}$ can be aligned with the current frame $I_{t}$ by the perspective transformation.

$$\hat{I}_{t-k} = \emph{H} I_{t-k} \tag{1}$$

Here, $H$ represents the perspective transformation matrix between $I_{t-k}$ and $I_{t}$, $\hat{I}_{t-k}$ denotes the motion-compensated previous $k$ frame.

#### III-A2. Motion feature extraction

After we have obtained the motion-compensated previous $k$ frame, we can highlight the moving areas with absolute differences between the current frame and the motion-compensated previous $k$ frame. Frame difference is a simple but effective technique to find the change, it can capture the pixel-level change between images. If we assume the acquisition time between neighboring frames is in a short range, we can expect that those differences originate from obviously moving objects. In general, there are many different versions of frame difference, such as 2-frame difference, 3-frame difference, and 5-frame difference. In this paper, a modified 3-frame difference is used to extract motion features. We first calculate the frame difference between two adjacent frames. The frame difference is defined as follows:

$$E_{1,2} = |I_{t}-\hat{I}_{t-k}| \tag{2}$$

$$E_{2,3} = |I_{t}-\hat{I}_{t+k}| \tag{3}$$

where $I_{t}$ is the gray value of the current frame, $\hat{I}_{t-k}$ and $\hat{I}_{t+k}$ are the gray value of the motion-compensated previous $k$ frame and next $k$ frame.

Next, based on extensive experiments, a threshold of 5 is applied on $E_{1,2}$ and $E_{2,3}$ to remove noise and highlight the silhouette of the potential moving objects. As a result, two binarized frame differences $D_{1,2}$ and $D_{2,3}$ are obtained. Finally, logical OR operation is applied on two binarized frame difference masks to enhance the motion feature $E_{t}$ as follows:

$$E_{t} = D_{1,2} \cup D_{2,3} \tag{4}$$

Different from the classical three-frame difference method that uses logical AND operation to suppress noise and obtain a more intact outline, logical OR operation can enhance motion features of tiny moving objects by aggregating moving areas from multiple frames. A detailed comparison between different frame difference methods will be given in Section IV-D.

#### III-A3. Post-processing

Frame difference can segment the pixels belonging to moving objects, but there are also many noise, small holes, and disconnected blobs. Hence, morphological operation and connected component analysis are used to obtain more precise bounding boxes. Firstly, the morphological open and close operation with a structure element of size 3×3 is used iteratively to eliminate isolated pixels and fill the holes. Then, connected component analysis is exploited to obtain the total number of pixels of each object, and the blob below 15 pixels is eliminated because such a small object usually belongs to interruption or alignment error. Finally, the minimum bounding rectangle is applied to mark the moving objects and obtain the final bounding boxes.

### III-B. Trajectory filtering

We have obtained the candidate moving objects after the motion feature enhancement. However, there are still many false positives such as cars, pedestrians, swaying trees and image alignment errors. Especially for image alignment errors, the false positives generated by motion parallax are severe when MAV flies at low altitudes within urban areas. To remove these false positives, we first adopt multi-object tracking to generate a trajectory for each object. Then, based on the motion feature analysis, we can eliminate these false positives by trajectory classification.

#### III-B1. Trajectory generation

To generate the trajectory for each candidate moving object, we adopt the classical multi-object tracking algorithm SORT [27]. The SORT uses a Kalman filter with a constant velocity motion model to predict the tracked objects' center, size, and aspect ratio in the current frame. Data association is viewed as a linear assignment problem that can be solved by the Hungarian algorithm [28]. If the intersection over union (IOU) between one detection and one tracking result is over 0.3, the detection is associated with that trajectory. Otherwise, a new tracking is activated. The max age of each trajectory is set to 3. It means that if one trajectory fails to be associated with any detection for successive three frames, the trajectory would be deleted. The min-hits are set to 1 to start a trajectory immediately after a successful detection.

#### III-B2. Trajectory classification

In the previous step, the trajectories of candidate moving objects have been generated with multi-object tracking. To remove trajectories belonging to image alignment errors and swaying trees, a trajectory classification algorithm based on motion analysis is proposed. In particular, the proposed trajectory classification model mainly considers the temporal and spatial features of candidate moving objects.

*1) Temporal feature:* By observing the motion feature of moving objects and false positives, we found that the movement of MAVs is usually continuous and lasts for several frames. However, the movement of false positives such as swaying trees and image alignment errors usually lasts for a short time. Therefore, we assign an object with a high probability of being a moving object if it moves for a long time. In particular, we view the object with an age over 3 frames as a potential moving object.

*2) Spatial feature:* On the other hand, we notice that the false positives usually move within a small range. As a comparison, real moving objects such as pedestrians, cars, and MAVs usually move over a wide range. Therefore, we give objects with large distances between consecutive frames with a high probability of being an MAV. In particular, the distance metric is defined as

$$D(b_{i},b_{j})=\frac{\rho(b_{i},b_{j})}{d(b_{i},b_{j})} \tag{5}$$

where $b_{i}$ and $b_{j}$ are bounding boxes of the same object in the $i$-th frame and $j$-th frame, $\rho(b_{i},b_{j})$ represents the central Euclidean distance between $b_{i}$ and $b_{j}$, $d(b_{i},b_{j})$ represents the diagonal length of the smallest enclosing box of $b_{i}$ and $b_{j}$.

*3) Trajectory Classification:* The details of the algorithm calculating the probability of each target being a potential moving target or a false positive are given in Algorithm 1. For each target, we define $S=\{s_{1},s_{2},...,s_{N}\}$ as the set of all the bounding boxes of the target over $N$ frames. We first align $s_{N-3}$ to the frame of $s_{N}$ by the previously mentioned perspective transformation. Then, we calculate the distance metric between $s_{N-3}$ and $s_{N}$ and classify the target as a moving object or a false positive directly based on the value of the distance metric. An example of trajectory filtering is demonstrated in Fig. 4. As we can see, the image alignment errors such as lampposts and trees are mostly removed, and the true moving objects such as MAVs, cars, and pedestrians are reserved.

**Algorithm 1** The trajectory classification algorithm

```
Input:  The bounding box of a potential moving target, m_i;
        The set of stored N bounding boxes of the potential moving target, S={s_1, s_2, ..., s_N};
Output: The probability of a potential moving target, P;

1:  Define f(·) a function counting the size of a set;
2:  Put m_i into S at the last position;
3:  if f(S) ≥ 3 then
4:      Align the s_{N-3} to the frame of s_N;
5:      Calculate D(s_{N-3}, s_N);
6:      P = { 1, if D(s_{N-3}, s_N) ≥ 1;
            0, if D(s_{N-3}, s_N) < 1 };
7:  else
8:      P = 0;
9:  end if
```

![Fig.4(a) Before trajectory filtering](assets/fig4_1.png)

![Fig.4(b) After trajectory filtering](assets/fig4_2.png)

**Fig. 4:** An example of trajectory filtering. After the trajectory filtering, most of the bounding boxes generated by image alignment errors are removed.

### III-C. Local fine detection

After the trajectory filtering, the candidate moving objects still contain some false positives such as moving cars, pedestrians, flying birds, and a few image alignment errors. Moreover, the bounding boxes generated by motion feature enhancement are usually redundant with excessive background areas. Therefore, we apply a local appearance-based classifier (LAC) and a local appearance-based detector (LAD) to further eliminate false positives and obtain a refined bounding box.

#### III-C1. Local appearance-based classifier

The false positives such as moving cars, pedestrians, and flying birds usually have similar motion features but different appearance features with MAV, we can use an appearance-based classifier to classify them into MAV and clutter. Considering that the appearance-based classifier will be called frequently and most of the common CNN architectures have a very deep structure that requires large computation resources, a three-layer CNN network is used to extract the feature of MAV and classify candidate moving objects into MAV and clutter. We train the proposed CNN model with local images of candidate moving objects. The input images are resized to 32×32 before being sent to the network. Each convolution layer uses ReLU activation and max pooling. The final layer uses a fully connected work with a softmax activation to classify the candidate moving objects into two classes.

#### III-C2. Local appearance-based detector

In the previous step, we obtained the candidate MAV targets with an appearance-based classifier. However, since some targets are too small to recognize and a shallow network can hardly extract enough high-level semantic information from the MAV target, the candidate MAV targets still contain some false positives that look similar to tiny MAVs. In addition, the bounding boxes generated through morphological operation are not accurate enough and have redundant enclosing regions. Therefore, an appearance-based detector is applied on the local region cropped from the neighboring area of the candidate MAV targets to obtain the precise position and final detection result.

In this paper, a 320×320 area around the center of the candidate MAV target is cropped as the local search region. Then, YOLOv5s is applied to the cropped image for refined MAV detection. YOLO is a fast single-shot object detector based on convolutional neural networks. Our previous study [5] shows that YOLO can achieve a good balance between accuracy and speed. To improve its performance on small object detection, a small object detection layer is added to concatenate the shallow feature map with the deep feature map. The local detector is trained with cropped images from the training dataset and the target with a confidence over 0.5 is viewed as a successful detection result.

## IV. Experiments

### IV-A. Datasets

Most of the existing MAV datasets are either collected by stationary cameras such as [4], [6] or have relatively large object sizes and simple backgrounds [7]. As far as we know, there are few datasets specifically designed for small MAV detection under complex and non-planar scenes from an aerial view. To evaluate the performance of the proposed MGMD algorithm, we test our algorithm on our previously proposed ARD-MAV dataset[^1]. This dataset contains 60 video sequences and 106,665 frames. All the videos are taken by the cameras of DJI Mavic2 or DJI M300 flying at low and medium altitudes. The videos contain many real-world challenges such as complex backgrounds, non-planar scenes, occlusion, abrupt camera movement, fast-moving MAVs, and small MAVs. Each video is about one minute long with a 30 FPS frame rate and has a resolution of 1920×1080. The object size ranges from 6×3 to 136×75. The average object size is only 0.02% of the image size. As far as we know, this is the smallest average object size among the existing MAV datasets. We use 50 videos for training and 10 videos for testing. In particular, the test videos are mainly composed of small MAVs and MAVs under complex backgrounds.

[^1]: https://github.com/WestlakeIntelligentRobotics/Global-Local-MAV-Detection

### IV-B. Metrics and implementation details

**Metrics.** Following the related works [12], in this experiment, the performance evaluation is based on commonly used metrics such as precision, recall, F1-Score, and average precision (AP). Since the targets in this dataset are too small and a small deviation of the position of the bounding box will result in a large change of intersection over union (IOU), we set the IOU threshold between predictions and ground truth to 0.25. Therefore, detected targets matching with ground truth with IOU > 0.25 are counted as true positives.

**Implementation details.** Our experiments are implemented on a computer with an Intel i7 processor, 32GB RAM, and an NVIDIA Geforce RTX 3070 GPU. For the training of the local fine detector, the input image size is set to 320×320. We use the Adam optimizer with a momentum of 0.937 and an initial learning rate of 0.01. The training is started from publicly available pre-trained model weights of YOLOv5s on MS-COCO. For the appearance-based classifier, the Adam optimizer is applied with a learning rate of 0.001. We trained the model for 100 epochs with a batch size of 64.

### IV-C. Comparison with state-of-the-art methods

We compare our proposed algorithm with recently published methods such as one-stage object detector, aerial object detector [16], video object detector [17], and aerial small object detector [12]. In particular, to avoid the resolution loss introduced by the down-sampling method, YOLOv5s, TPH-YOLOv5s, and TPH-YOLOv5l use the 1536×1536 input image size for inference rather than the default 640×640. All compared methods are implemented based on official codes and fine-tuned using the pre-trained weights available with public codes.

The quantitative comparison of MGMD with the compared methods on the ARD-MAV dataset is shown in Table I. The experimental results demonstrate that our proposed algorithm outperforms the existing methods on various metrics. Specifically, our proposed method outperforms the best prior method by 19% on the AP metric. In addition, our method can run with 28 frames per second (FPS).

The qualitative comparison of MGMD with the compared methods is shown in Fig. 5. As we can see, the proposed method can detect very tiny MAVs and MAVs under complex backgrounds. However, the compared method can hardly detect a target MAV under such challenging conditions. It is important to note that the small MAV here denotes the MAV with a size smaller than 10×10 pixels, which is difficult even for humans to recognize in the image.

**TABLE I:** Quantitative comparison of the MGMD with state-of-the-art methods on ARD-MAV dataset.

| Method | P | R | F1 | AP | FPS |
|---|---|---|---|---|---|
| YOLOv5s | 0.86 | 0.27 | 0.41 | 0.26 | **82** |
| TPH-YOLOv5s [16] | 0.81 | 0.30 | 0.44 | 0.34 | 35 |
| TPH-YOLOv5l [16] | **0.99** | 0.32 | 0.48 | 0.36 | 11 |
| Dogfight [12] | 0.77 | 0.24 | 0.36 | 0.24 | 1 |
| MEGA [17] | 0.26 | 0.37 | 0.31 | 0.23 | 3 |
| MGMD | 0.84 | **0.59** | **0.69** | **0.55** | 28 |

![Fig.5 qualitative comparison](assets/fig5_1.png)

![Fig.5 qualitative comparison (2)](assets/fig5_2.png)

![Fig.5 qualitative comparison (3)](assets/fig5_3.png)

**Fig. 5:** The qualitative comparison between the detection results from TPH-YOLOv5l, MEGA, and MGMD. MGMD can detect the target MAV when the appearance features are unreliable under challenging conditions. **Yellow box** indicates the target detected by our proposed method. **Red box** indicates the target detected by TPH-YOLOv5l. **Blue box** indicates the target detected by MEGA. Because the image size is too large compared to the target's size, we only show the *cropped regions* from the full-size image for a better view.

### IV-D. Ablation studies

To examine the effectiveness of different components in MGMD, we conduct extensive experiments to study how they contribute to the final performance.

**Motion Feature Enhancement.** We have tested different frame difference methods such as 2-frame difference, and classical 3-frame difference to verify the effectiveness of our proposed motion feature enhancement module. The experimental results in Table II show that our proposed 3-frame difference with OR operation can effectively improve the recall while sacrificing a small amount of precision and inference speed.

**TABLE II:** Ablation study of different frame difference methods.

| Method | P | R | F1 | FPS |
|---|---|---|---|---|
| 2-frame | 0.82 | 0.45 | 0.58 | **43** |
| 3-frame + AND | **0.87** | 0.46 | 0.61 | 33 |
| 3-frame + OR | 0.84 | **0.59** | **0.69** | 28 |

**Trajectory Filtering.** This paper mainly uses the TF module to remove false positives such as swaying trees and image alignment errors in non-planar scenes. As shown in Table III, when the TF module is introduced, the precision and inference speed are improved in most cases. However, the TF module sometimes reduces the recall as it mistakenly removes the hovering or slowly moving MAVs.

**Local Appearance-based Classifier.** As shown in the first and third row of Table III, the LAC module significantly improves detection accuracy and inference speed. This is because the LAC module can effectively remove interruptions with similar motion features but looks distinctly different from MAV. As a result, the workload of the LAD module is greatly reduced and the detection accuracy and inference speed are improved.

**Local Appearance-based Detector.** As shown in the second and third row of Table III, the LAD plays an important role by promoting the precision metric. Since the LAC works on the segmented images generated by motion feature enhancement and is trained with a shallow network, it can hardly discriminate false positives that look similar to tiny MAVs and have a coarse bounding box. However, the LAD is trained with ground-truth labels, it has a bounding box refinement module and better semantic representation towards MAVs. Therefore, it can significantly improve the detection accuracy by eliminating false positives that look similar to tiny MAVs.

**TABLE III:** Ablation study of different components in MGMD.

| Method | P | R | F1 | FPS |
|---|---|---|---|---|
| MFE + LAD | 0.60 | 0.58 | 0.59 | 16 |
| MFE + LAC | 0.40 | 0.39 | 0.39 | **34** |
| MFE + LAD + LAC | 0.79 | **0.63** | **0.70** | 23 |
| MFE + TF + LAD | 0.79 | 0.58 | 0.67 | 22 |
| MFE + TF + LAC | 0.45 | 0.37 | 0.41 | 32 |
| MFE + TF + LAC + LAD | **0.84** | 0.59 | 0.69 | 28 |

### IV-E. Discussion

Experimental results demonstrate that our proposed method outperforms the state-of-the-art methods on various metrics. We attribute it to several factors. First, MGMD adopts a motion feature enhancement module to extract motion information and can detect the target MAV when the target is extremely small or the background is complex. However, appearance-based detectors such as YOLOv5 and TPH-YOLOv5 fail to capture the appearance feature in these scenarios. In addition, the existing motion-based methods such as [12] also have difficulties under such challenging conditions. Some examples are shown in Fig. 6. Second, MGMD uses a local fine detector which works on a local search region cropped from the full-size image. This local search region greatly retains the valuable appearance information of small targets and eliminates the influence of other interruptions that look similar. Therefore, it achieves better performance on the precision metric. Third, the multi-object tracking method makes good use of the temporal feature of MAV and eliminates the interruptions that occasionally emerge.

![Fig.6(a) The input image](assets/fig6_a.png)

![Fig.6(b) Heatmap of YOLOv5](assets/fig6_b.png)

![Fig.6(c) Motion boundary by optical flow](assets/fig6_c.png)

![Fig.6(d) Motion feature by our method](assets/fig6_d.png)

**Fig. 6:** A comparison of the appearance feature and motion feature generated by different methods. (a) The input image. (b) Heatmap of YOLOv5. (c) The motion boundary extracted by optical flow gradient [12]. (d) The motion feature extracted by our method.

## V. Conclusion

In this paper, we propose a simple yet effective algorithm for small MAV detection in complex and non-planar scenes. Our main contribution is to extract the motion features of extremely small MAVs using the proposed motion feature enhancement module, then utilize a trajectory-based classifier and an appearance-based classifier to eliminate false positives generated by motion parallax and other moving objects. In addition, a modified YOLOv5s is applied to the cropped regions to achieve refined detection results. Ablation experiments show the effectiveness of each module in our method. Together, our proposed algorithm could achieve an excellent performance of 0.55 AP on the ARD-MAV dataset which contains various challenging conditions, and achieve a running speed of 28 FPS.

Although the proposed method has made some progress, it still faces the problem of false detections when detecting small targets with high similarity to MAVs such as tiny birds and cars. In addition, our proposed method mainly focuses on moving targets, the hovering and slowly moving MAVs are usually ignored. In the future, we will try to extend our algorithm to both moving and hovering MAVs. Moreover, an end-to-end network that can learn motion cues and appearance features within a unified network is necessary to be designed to simplify the training process and reduce the empirical parameters.

## Acknowledgments

This should be a simple paragraph before the References to thank those individuals and institutions who have supported your work on this article.

## References

[1] Y. Tang, Y. Hu, J. Cui, F. Liao, M. Lao, F. Lin, and R. S. Teo, "Vision-aided multi-UAV autonomous flocking in GPS-denied environment," *IEEE Transactions on Industrial Electronics*, vol. 66, no. 1, pp. 616–626, 2018.

[2] J. Park and A. J. Choi, "Vision-based in-flight collision avoidance control based on background subtraction using embedded system," *Sensors*, vol. 23, no. 14, p. 6297, 2023.

[3] F. Samadzadegan, F. Dadrass Javan, F. Ashtari Mahini, and M. Gholamshahi, "Detection and recognition of drones based on a deep convolutional neural network using visible imagery," *Aerospace*, vol. 9, no. 1, pp. 31–51, 2022.

[4] B. K. S. Isaac-Medina, M. Poyser, D. Organisciak, C. G. Willcocks, T. P. Breckon, and H. P. H. Shum, "Unmanned aerial vehicle visual detection and tracking using deep neural networks: A performance benchmark," in *Proceedings of the 2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)*, pp. 1223–1232, 2021.

[5] Y. Zheng, Z. Chen, D. Lv, Z. Li, and S. Zhao, "Air-to-air visual detection of micro-UAVs: An experimental evaluation of deep learning," *IEEE Robotics and Automation Letters*, vol. 6, no. 2, pp. 1020–1027, 2021.

[6] J. Zhao, J. Zhang, D. Li, and D. Wang, "Vision-based anti-UAV detection and tracking," *IEEE Transactions on Intelligent Transportation Systems*, vol. 23, no. 12, pp. 25323–25334, 2022.

[7] J. Li, D. H. Ye, M. Kolsch, J. P. Wachs, and C. A. Bouman, "Fast and robust UAV to UAV detection and tracking from video," *IEEE Transactions on Emerging Topics in Computing*, vol. 10, no. 3, pp. 1519–1531, 2021.

[8] L. Du, C. Gao, Q. Feng, C. Wang, and J. Liu, "Small UAV detection in videos from a single moving camera," in *Proceedings of the 2017 CCCV Communications in Computer and Information Science (CCIS)*, pp. 187–197, 2017.

[9] C. Wang, T. Wang, E. Wang, E. Sun, and Z. Luo, "Flying small target detection for anti-UAV based on a gaussian mixture model in a compressive sensing domain," *Sensors*, vol. 19, no. 9, pp. 2168–2182, 2019.

[10] Y. Zheng, C. Zheng, X. Zhang, F. Chen, Z. Chen, and S. Zhao, "Detection, localization, and tracking of multiple MAVs with panoramic stereo camera networks," *IEEE Transactions on Automation Science and Engineering*, vol. 20, no. 2, pp. 1226–1243, 2022.

[11] J. Xie, C. Gao, J. Wu, Z. Shi, and J. Chen, "Small low-contrast target detection: Data-driven spatiotemporal feature fusion and implementation," *IEEE Transactions on Cybernetics*, vol. 52, no. 11, pp. 11847–11858, 2022.

[12] M. W. Ashraf, W. Sultani, and M. Shah, "Dogfight: Detecting drones from drones videos," in *Proceedings of the 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 7063–7072, 2021.

[13] H. Wang, X. Wang, C. Zhou, W. Meng, and Z. Shi, "Low in resolution, high in precision: UAV detection with super-resolution and motion information extraction," in *Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pp. 1–5, IEEE, 2023.

[14] Z. Teed and J. Deng, "Raft: Recurrent all-pairs field transforms for optical flow," in *Proceedings of the 16th European Conference on Computer Vision (ECCV)*, pp. 402–419, Springer, 2020.

[15] P. Zhang, J. Zhao, C. Bo, D. Wang, H. Lu, and X. Yang, "Jointly modeling motion and appearance cues for robust RGB-T tracking," *IEEE Transactions on Image Processing*, vol. 30, pp. 3335–3347, 2021.

[16] X. Zhu, S. Lyu, X. Wang, and Q. Zhao, "TPH-YOLOv5: Improved YOLOv5 based on transformer prediction head for object detection on drone-captured scenarios," in *Proceedings of the 2021 IEEE/CVF International Conference on Computer Vision (ICCV) Workshops*, pp. 2778–2788, October 2021.

[17] Y. Chen, Y. Cao, H. Hu, and L. Wang, "Memory enhanced global-local aggregation for video object detection," in *Proceedings of the 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 10334–10343, 2020.

[18] C. Rui, G. Youwei, Z. Huafei, and J. Hongyu, "A comprehensive approach for UAV small object detection with simulation-based transfer learning and adaptive fusion," *arXiv preprint arXiv:2109.01800*, 2021.

[19] A. Rozantsev, V. Lepetit, and P. Fua, "Detecting flying objects using a single moving camera," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 39, no. 5, pp. 879–892, 2016.

[20] T. Sangam, I. R. Dave, W. Sultani, and M. Shah, "Transvisdrone: Spatio-temporal transformer for vision-based drone-to-drone detection in aerial videos," in *Proceedings of the 2023 IEEE International Conference on Robotics and Automation (ICRA)*, pp. 6006–6013, IEEE, 2023.

[21] X. Wang, Y. Yan, H. Sun, and D. Zhu, "Dense-and-similar object detection in aerial images," *Pattern Recognition Letters*, vol. 176, pp. 153–159, 2023.

[22] C. Yang, Y. Cao, and X. Lu, "Towards better small object detection in UAV scenes: Aggregating more object-oriented information," *Pattern Recognition Letters*, vol. 182, pp. 24–30, 2024.

[23] R. LaLonde, D. Zhang, and M. Shah, "Clusternet: Detecting small objects in large scenes by exploiting spatio-temporal information," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 4003–4012, 2018.

[24] C. W. Corsel, M. van Lier, L. Kampmeijer, N. Boehrer, and E. M. Bakker, "Exploiting temporal context for tiny object detection," in *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, pp. 79–89, 2023.

[25] S. Minaeian, J. Liu, and Y.-J. Son, "Effective and efficient detection of moving targets from a UAV's camera," *IEEE Transactions on Intelligent Transportation Systems*, vol. 19, no. 2, pp. 497–506, 2018.

[26] M. Ellenfeld, S. Moosbauer, R. Cardenes, U. Klauck, and M. Teutsch, "Deep fusion of appearance and frame differencing for motion segmentation," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, pp. 4339–4349, 2021.

[27] A. Bewley, Z. Ge, L. Ott, F. Ramos, and B. Upcroft, "Simple online and realtime tracking," in *Proceedings of the 2016 IEEE International Conference on Image Processing (ICIP)*, pp. 3464–3468, 2016.

[28] H. W. Kuhn, "The hungarian method for the assignment problem," *Naval research logistics quarterly*, vol. 2, no. 1-2, pp. 83–97, 1955.
