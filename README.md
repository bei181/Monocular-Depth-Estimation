# Monocular Depth Estimation

## Papers of CVPR2021  

**[1] MonoIndoor: Towards Good Practice of Self-Supervised Monocular Depth Estimation for Indoor Environments**  
**[paper](https://arxiv.org/abs/2107.12429)**  
内容：从深度分解模块和姿态残差估计模块进行室内场景的深度估计。针对室内场景的痛点（深度范围变化大&较大的旋转运动），提出解决方法。位姿预测采用的是残差姿态估计，基于估计的初始位姿。并且，还利用自注意力模块去预测了场景的尺度因子。

**[2] Regularizing Nighttime Weirdness: Efficient Self-supervised Monocular Depth Estimation in the Dark**  
**[paper](https://arxiv.org/abs/2108.03830)**  
内容：旨在解决低能见度和暗光场景下的深度估计方法，文章提出了三个点：使用了正则化方法学习先验分布；并基于映射一致的图像增强模块提升图像对比度和可见度；基于统计的mask策略以去除弱纹理区域带来的干扰

**[3] Towards Interpretable Deep Networks for Monocular Depth Estimation**  
**[paper](https://arxiv.org/abs/2108.05312) | [code](https://github.com/youzunzhi/InterpretableMDE)**  
内容: 单目深度估计的可解释性问题，方便对不同的数据集，模型进行适用性分析和误差模型。
主要贡献：（1）基于模型内部单元的深度选择性量化了 MDE 深度网络的可解释性； （2）提出了一种新的方法来为 MDE 学习可解释的深度网络，而无需修改原始网络的架构或需要任何额外的注释； （3）有效地提高了深度 MDE 网络的可解释性，同时不会损害甚至提高深度精度

**[4] Self-supervised Monocular Depth Estimation for All Day Images using Domain Separation**  
**[paper](https://arxiv.org/abs/2108.07628)**   
内容：提出了一个域分离(domain-separated)网络，用于全天图像的自我监督深度估计。具体来说，将昼夜图像对的信息划分为两个互补的子空间：私有域和不变域，其中前者包含特有信息（照度等）的白天和黑夜图像，后者包含重要的共享信息（纹理等）。为了保证白天和黑夜图像包含相同的信息，夜间图像由白天图像和GAN 生成。

**[5] StructDepth: Leveraging the structural regularities for self-supervised indoor depth estimation**  
**[paper](https://arxiv.org/abs/2108.08574) | [code](https://github.com/SJTU-ViSYS/StructDepth)**  
内容：这个方法属于室内场景的自监督单目深度估计，加入了两个额外的监督信号：1) 曼哈顿法线约束和 2) 共面约束。曼哈顿法线约束使表面与主要方向对齐，共面约束是针对同一平面区域内的3D点，是平面可以更好地拟合它们。 

**[6]Fine-grained Semantics-aware Representation Enhancement for Self-supervised Monocular Depth Estimation**  
**[paper](https://arxiv.org/abs/2108.08829) | [code](https://github.com/hyBlue/FSRE-Depth)**  
内容：为了克服光度一致性在低纹理区域和物体边界处监督效果差的问题，论文通过利用场景语义信息来改进自监督单目深度估计。将隐式语义知识整合到几何表示增强中，并提出一种利用语义引导的局部几何来优化中间深度表示的度量学习方法，以及一种新颖的利用两个异构特征表示之间的跨模态的新型特征融合模块。

**[7] Depth360: Monocular Depth Estimation using Learnable Axisymmetric Camera Model for Spherical Camera Image**  
**[paper](https://arxiv.org/pdf/2110.10415.pdf)**  
内容：针对鱼眼相机模型的单目深度估计，而且相机模型(内参系数)是靠网络自己学习的。同样也做了KITTI数据集上针孔相机模型的实验。

**[8] PlaneRecNet: Multi-Task Learning with Cross-Task Consistency for Piece-Wise Plane Detection and Reconstruction from a Single RGB Image**  
**[paper](https://arxiv.org/pdf/2110.11219.pdf)**  
内容：实例分割和深度估计融合来做，提出了几个新的损失函数（几何约束），共同提高了分段平面分割和深度估计的准确性。还添加了一种新颖的平面先验注意力模块用于指导具有平面实例意识的深度估计。

**[9] Pseudo Supervised Monocular Depth Estimation with Teacher-Student Network**  
**[paper](https://arxiv.org/pdf/2110.11545.pdf)**  
内容：双目训练，单目测试。利用无监督双目深度估计方法训练教师模型，训练时还利用了语义分割真值标签。随后，利用教师模型给学生模型生成深度图伪标签，学生模型是有监督的单目深度估计网络。学生模型的loss中还加入了语义引导的平滑loss和利用遮挡map的重构loss。

**[10] Absolute distance prediction based on deep learning object detection and monocular depth estimation models**  
**[paper](https://arxiv.org/ftp/arxiv/papers/2111/2111.01715.pdf)**  
内容：将目标检测和深度估计合在一起做，单目深度估计用的原理就是SFMLearner一样的自监督思想，就是在decoder里加了GCN(图卷积)。最后尺度恢复拟合了一个二次函数模型。

**[11] CamLessMonoDepth: Monocular Depth Estimation with Unknown Camera Parameters**  
**[paper](https://arxiv.org/pdf/2110.14347.pdf)**  
内容：提出一种仅从单目图像序列中学习来隐式估计针孔相机内参以及深度和姿态的方法。通过利用有效的子像素卷积，获得了高保真深度估计，网络中还加入了像素级不确定性估计。
 
**[12] Self-Supervised Monocular Scene Decomposition and Depth Estimation**  
**[paper](https://arxiv.org/pdf/2110.11275.pdf)**   
内容：提出MonoDepthSeg, 在不使用任何真实标签的情况下联合估计深度并从单目视频中分割移动对象。 将场景分解为固定数量的component，其中每个组件对应于图像上的一个区域，并且拥有其自身的变换矩阵，表示其运动。

**[13] Self-Supervised Monocular Depth Estimation with Internal Feature Fusion**  
**[paper](https://arxiv.org/pdf/2110.09482.pdf)**   
基于成熟的语义分割网络 HRNet，提出了一种新颖的深度估计网络 DIFFNet，它可以在下采样和上采样过程中利用语义信息，并且采用了特征融合和注意力机制。

**[14] Residual-Guided Learning Representation for Self-supervised Monocular Depth Estimation**  
**[paper](https://arxiv.org/pdf/2111.04310.pdf)**   
对Featdepth改进，提出了一个新的损失Residual-Guidance Loss。作者认为，autoencoder和DepthNet是来自不同的loss，并没有完全发掘DepthNet网络需要的的特征。所以，作者提出这个loss，可以让DepthNet引导autoencoder一起学习，效果略微有提升。

**[15] Attention meets Geometry: Geometry Guided Spatial-Temporal Attention for Consistent Self-Supervised Monocular Depth Estimation**  
**[paper](https://arxiv.org/pdf/2110.08192.pdf)**    
提出了一个空间注意力模块，将粗略的深度预测与聚合局部几何信息相关联，以及一种新颖的时间注意力机制，进一步在全局上下文中处理连续图像中的局部几何信息。

**[16] PLNet: Plane and Line Priors for Unsupervised Indoor Depth Estimation**  
**[paper](https://arxiv.org/pdf/2110.05839.pdf) | [code](https://github.com/HalleyJiang/PLNet)**   
室内场景充满了特定的结构，例如平面和线，有助于指导无监督的深度估计学习。本文提出了利用平面和线先验来增强深度估计的 PLNet，使用局部平面系数表示场景几何并对表示施加平滑约束，以及，随机选择一些可能共面或共线的点集来强制执行平面和线性一致性。

**[17]Monocular Depth Estimation with Sharp Boundary**  
**[paper](https://arxiv.org/pdf/2110.05885.pdf)**  
为了缓解边缘模糊问题，设计了一个场景理解模块来学习具有低层和高层特征的全局信息，以及一个尺度变换模块将全局信息转换为不同的尺度。作者还提出了一个边界感知深度损失函数

**[18]ManyDepth-The Temporal Opportunist: Self-Supervised Multi-Frame Monocular Depth**  
**[paper](https://arxiv.org/pdf/2104.14540.pdf) | [code](https://github.com/nianticlabs/manydepth)**  
提出了一种基于深度端到端cost volume的方法，以及一种新颖的一致性损失，鼓励网络在被认为不可靠时忽略cost volume，例如在移动物体的情况下，以及处理静态相机的增强方案。




