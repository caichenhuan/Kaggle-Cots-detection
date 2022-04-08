# Kaggle-Cots-detection

使用目标检测的方法来检测海底视频中的海星。

比赛链接： [Kaggle: TensorFlow - Help Protect the Great Barrier Reef](https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/overview)



## 赛题简介

### 研究背景

由于以珊瑚为食的棘冠海星（COTS）的数量过多，威胁到了澳大利亚珊瑚礁以及生态平衡，于是科学家、旅游经营者和珊瑚礁管理者准备了一个大规模干预计划，将棘冠海星（COTS）的数量控制在生态可持续的水平里。

### 研究任务

通过建立一个在珊瑚礁水下视频上训练的物体检测模型，实时准确地识别测试图片中的海星。

### 评估指标

此比赛根据 `F2-Score` 来计算分数，预测框与真实框的 `IoU` 值达到阈值则为正确识别，同时该指标以 0.05 的步长扫描 0.3 到 0.8 范围内的 `IoU` 阈值，计算每个阈值的 `F2-Score`  分数，最终求和平均的值即为提交分数。使得更多的海星被成功预测同时容忍一些误判，可以提高  分`F2-Score` 数。

<img src="assets\readme\formula_fscore.png" alt="formula_fscore" style="zoom:80%;" />

## 数据集简介

数据集来源：[The CSIRO Crown-of-Thorn Starfish Detection Dataset](https://arxiv.org/abs/2111.14311)

训练数据集是 3 段视频，包括 23503 张 1280 x 720 的图片，以及包含图片中的目标框的 train.csv 文件。如图是将目标框标识在图片中的展示。

- `train_images/` - 训练数据集，形如 `video_{video_id}/{video_frame}.jpg`.
- `annotations` - 储存 Python 字符串格式的海星检测框的数据，在 test.csv 中不可用，描述了边界框由图像左下角的像素坐标`(x_min, y_min)`及其以像素为单位的`width`和`height`，（COCO格式）

<img src="assets\readme\image-1.png" alt="image-1" style="zoom:80%;" />

picture-1（包含目标框）

<img src="assets\readme\image-2.png" alt="image-2" style="zoom:80%;" />

picture-2（包含目标框）

<img src="assets\readme\traincsv.png" alt="traincsv" style="zoom:80%;" />

train.csv部分



## 方案

### 1. 数据处理

训练数据集总共 23501 张图片，其中 20.93% 是有BBox的，因为没有BBox的图片都是背景，对模型训练并没有什么正向作用，所以去除没有BBox的图片，剩下的作为我们的训练数据集。

训练数据集分为 3 个视频，因为视频前后帧的强相关性，就不能随机拆分训练集和验证集，所以使用 3 折交叉验证，以不同的视频来划分。

将原始边缘框格式重写为 YOLO 格式。YOLO 格式：一个`.txt`对应一张图片，一行对应一个物体，第一部分是`class_id`，后面四个数分别是`BBox`的 (中心x坐标，中心y坐标，宽，高)，这些坐标都是 0～1 的相对坐标。将`.txt`的文件保存在`Kaggle/labels`的路径下。

### 2. 数据分析

对数据集中的 BBox 的数据分布进行可视化，对后续的策略能有一定的启发。

首先对检测框的中心位置进行可视化，如下图，可以看到检测框在 y 轴分布较均匀，在 x 轴的中间和中间偏左比较集中。

<img src="assets\readme\position-xy.png" alt="position-xy" style="zoom:80%;" />

pic.4 检测框中心位置


接下来对检测框的大小进行分析，如下图，可以观察到检测框的大小集中在 `20 x 20 ~ 60 x 60`左右，部分大的可以达到`200 x 200`的像素，总体来说尺寸较小。

<img src="assets\readme\lenth-bbox.png" alt="lenth-bbox" style="zoom:80%;" />

pic.5 检测框的长度和宽度


### 3. 训练策略

- **baseline**：训练了 1280 分辨率的 yolov5s 模型，使用默认参数，得到 `F2-score` 0.546。

- **高分辨率**：由于检测目标尺寸较小，使用更大的分辨率可以在一定程度上提高分数，于是尝试训练了 1280、2560、3000、3500 分辨率的 yolov5 模型，同时也使用不同分辨率来推理，最后在 3000 推理分辨率的情况下得到一下的测试结果（线上 `F2-score`）：

  | **YOLOv5s分辨率选择** | **F2-Scores** |
  | :-------------------- | ------------- |
  | 1280                  | 0.573         |
  | 2560                  | 0.583         |
  | **3000**              | **0.588**     |
  | 3500                  | 0.587         |

  同时考虑线上和线下表现得情况下，3000 分辨率训练的 yolov5s，3000 分辨率推理可以达到一个较高的分数。

- **数据增强**：修改数据增强的方式，尝试多种不同的参数与增强方法，实验发现，增加水平竖直翻转和随机旋转对分数有提高，同时设置 mosaic 0.25，mixup 0.5，scale 0.5，使得线上和线下的分数有了一点提高（+0.01）。

- **设置置信度阈值**：默认的置信度阈值是 0.6，其实可以更低，本次比赛的评分标准是`F2-Score`，这意味着识别出更多的正类比减少更多的错误预测更重要，在yolov5中设置`F2-Score`的计算，同时实验了不同的置信度阈值，得到阈值为 0.1 的时候线下和线上分数都相对更高。

- **Tracking**：考虑到预测视频的前后帧是强相关的，所以在预测时加入了 tracking 的方法。比较常用的有 Deepsort 方法，是利用匈牙利算法和卡尔曼滤波的一种 tracking 方法，在本次比赛中最终使用的时 Norfair 的方法，原理和 Deepsort大同小异，Norfair 可以使用欧几里得距离来查找轨迹。具体 Norfair 库的说明请看官方说明 [Norfair - GitHub](https://github.com/tryolabs/norfair) 

- **WBF**：WBF 即加权框融合算法，和 NMS 类似是一种去除冗余框并融合框的方法，但根据算法的不同这两种方法得到的效果也不同，WBF 算法虽然更耗时，但在做多模型合并预测结果的时候有更好的结果，下图为 WBF 和 NMS 的一点区别，WBF 使用了所有的预测框来共同计算得到最终的框，NMS 是丢弃了一些预测框。

  <img src="assets\readme\WBF.png" alt="WBF" style="zoom:80%;" />

- **模型融合**：考虑到多尺度以及稳定性，于是使用模型融合，最终使用一个 1280 分辨率和一个 3000 分辨率的 yolov5 模型。

  | **模型及组合**                       | **F2-Score** |
  | ------------------------------------ | ------------ |
  | 3000yolov5s                          | 0.601        |
  | 3000yolov5s + Tracking               | **0.625**    |
  | 2560yolov5s + 3000yolov5s + Tracking | 0.633        |
  | 1280yolov5s + 3000yolov5s + Tracking | 0.658        |
  | 1280yolov5l + 3000yolov5s + Tracking | **0.664**    |



## 其他说明

### repo说明

- 训练代码：`train.ipynb`
- 推理代码：`infer.ipynb`

### Reference

[https://www.kaggle.com/code/awsaf49/great-barrier-reef-yolov5-train](https://www.kaggle.com/code/awsaf49/great-barrier-reef-yolov5-train)

[https://www.kaggle.com/code/awsaf49/great-barrier-reef-yolov5-infer](https://www.kaggle.com/code/awsaf49/great-barrier-reef-yolov5-infer)

[https://arxiv.org/pdf/1910.13302.pdf](https://arxiv.org/pdf/1910.13302.pdf)

[https://github.com/tryolabs/norfair](https://github.com/tryolabs/norfair)

[https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/discussion/300638](https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/discussion/300638)





