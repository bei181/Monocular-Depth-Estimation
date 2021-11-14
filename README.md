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

**[19]Adversarial Domain Feature Adaptation for Bronchoscopic Depth Estimation**  
**[paper](https://arxiv.org/pdf/2109.11798.pdf)**  
医疗图像上的应用，支气管镜中的深度估计。利用合成数据先有监督地训练网络，然后用对抗网络进行domain adaptation迁移到真实数据上，也属于无监督的单目深度估计方法。

**[20]Excavating the Potential Capacity of Self-Supervised Monocular Depth Estimation**  
**[paper](https://arxiv.org/pdf/2109.12484.pdf) | [code](https://github.com/prstrive/EPCDepth)**  
在不增加成本的情况下挖掘自监督单目深度估计的潜在能力，提出了：
(1)一种称为数据嫁接(data grafting)的新型数据增强方法，迫使模型探索更多线索来推断垂直图像位置之外的深度，
(2)探索性自蒸馏损失，提出一种新的后处理方法生成自蒸馏标签监督——选择性后处
(3)全尺寸网络，旨在赋予编码器深度估计任务的专业化并增强模型的表示能力

**[21]Weakly-Supervised Monocular Depth Estimation with Resolution-Mismatched Data**  
**[paper](https://arxiv.org/pdf/2109.11573.pdf)**  
提出了一种新的弱监督框架来训练单目深度估计网络以生成的高分辨率深度图，即输入是 HR彩色图像，而真值是低分辨率(LR)深度图。所提出的弱监督框架由共享权重单目深度估计网络和用于蒸馏的深度重建网络组成

**[22]Survey on Semantic Stereo Matching / Semantic Depth Estimation**  
**[paper](https://arxiv.org/pdf/2109.10123.pdf)**    
利用语义信息的立体匹配或者深度估计的一篇综述性论文

**[23]Monocular Depth Estimation Using Laplacian Pyramid-Based Depth Residuals (LapDepth)**  
**[code](https://github.com/tjqansthd/LapDepth-release)**  
目前KITTI有监督单目深度估计排行榜排第二，但没找到论文，github上有代码和视频讲解

**[24]Improving 360◦ Monocular Depth Estimation via Non-local Dense Prediction Transformer and Joint Supervised and Self-supervised Learning**   
**[paper](https://arxiv.org/pdf/2109.10563.pdf)**  
提出了一种通过结合有监督学习和自监督学习来实现的联合学习方案。每个学习的弱点得到补偿，从而导致更准确的深度估计。还提出了一个非局部融合块，它在重建深度时保留了视觉Transformer编码的全局信息。

**[25]On the Sins of Image Synthesis Loss for Self-supervised Depth Estimation**   
**[paper](https://arxiv.org/pdf/2109.06163.pdf)**  
提出了一个问题，但没有解决。作者认为图像合成的改进不需要深度估计的改进，相反，优化图像合成可能会导致主要预测目标——深度估计的性能出现分歧。论文将这种发散现象归因于源自数据的偶然不确定性。

**[26]Augmenting Depth Estimation with Geospatial Context**   
**[paper](https://arxiv.org/pdf/2109.09879.pdf)**  
一个新的思路，如果捕获位置是已知的，则相应的俯视图为理解场景的规模提供了宝贵的资源。论文提出了一种用于深度估计的端到端架构，该架构使用地理空间上下文从位于同一位置的俯视图推断合成的地面深度图，然后将其融合到编码器/解码器风格的分割网络中。

**[27]RVMDE: Radar Validated Monocular Depth Estimation for Robotics**   
**[paper](https://arxiv.org/pdf/2109.05265.pdf)**  
探索了来自雷达的粗信号与来自单目相机的细粒度数据融合，用于在恶劣环境条件下进行深度估计，有监督方法。

**[28]LiDARTouch: Monocular metric depth estimation with a few-beam LiDAR**   
**[paper](https://arxiv.org/pdf/2109.03569.pdf)**  
通过将单目相机与轻量级 LiDAR（4线）相结合，提高单目深度估计效果。LiDAR 输入作为额外的模型的输入，不仅在自监督 LiDAR 重建目标函数中使用，而且还用来估计姿势的变化。此外，该方法还减轻了仅相机方法所遭受的尺度模糊和无限深度问题。

**[29]Pano3D: A Holistic Benchmark and a Solid Baseline for 360o Depth Estimation**   
**[paper](https://arxiv.org/pdf/2109.02749.pdf)**  
Pano3D 是一种从球形全景图进行深度估计的新基准（benchmark）。 它旨在评估所有深度估计特征的性能，直接的深度估计性能（精度和准确性），以及次要特征(边界保留和平滑度)。 此外，Pano3D 超越了典型的数据集内评估，转向了数据集间性能评估。

**[30]Unsupervised Monocular Depth Perception: Focusing on Moving Objects**   
**[paper](https://arxiv.org/pdf/2108.13062.pdf)**  
提出了一种异常值masking技术，该技术将被遮挡或动态像素视为光度误差图中的统计异常值，使网络可以更准确地学习向相机相反方向移动的对象的深度。还提出了一种有效的加权多尺度方案，以减少预测深度图中的伪影。

**[31]Panoramic Depth Estimation via Supervised and Unsupervised Learning in Indoor Scenes**   
**[paper](https://arxiv.org/pdf/2108.08076.pdf)**  
以室内场景为重点的全景单目深度估计。扩展了 PADENet，适用于全景图像中不同尺度的失真。 同时，网络通过相应的模块学习全局和局部特征。

**[32]Monocular Depth Estimation Primed by Salient Point Detection and Normalized Hessian Loss**   
**[paper](https://arxiv.org/pdf/2108.11098.pdf)**  
源自显着点检测的自注意力机制，提出了一个准确且轻量级的单目深度估计框架。 ，利用一组稀疏的关键点来训练 FuSaNet 模型，该模型由两个主要组件组成：Fusion-Net 和 Saliency-Net。 此外，还引入了一个归一化的 Hessian 损失项，对沿深度方向的缩放和剪切保持不变。

**[33]DnD: Dense Depth Estimation in Crowded Dynamic Indoor Scenes**   
**[paper](https://arxiv.org/pdf/2108.05615.pdf)**  
一种无监督的在复杂而拥挤的室内环境中的单目深度估计方法。 通过对动态场景的训练来预测由静态背景和多个移动的人组成的整个场景的绝对比例深度图。 利用 RGB 图像和传统 3D 重建方法（SFM）生成的稀疏深度图来估计密集深度图，还使用了两个约束来处理非刚性移动的人的深度。

**[34]R4Dyn: Exploring Radar for Self-Supervised Monocular Depth Estimation of Dynamic Scenes**   
**[paper](https://arxiv.org/pdf/2108.04814.pdf)**  
使用雷达作为弱监督信号，通过过滤和扩展信号以使其与基于学习的方法兼容，解决了雷达固有的问题，例如噪声和稀疏性

**[35]CI-Net: Contextual Information for Joint Semantic Segmentation and Depth Estimation**   
**[paper](https://arxiv.org/pdf/2107.13800.pdf)**  
提出了一个注入上下文信息的网络（CI-Net）在编码器中引入了自注意力块来生成注意力图。在语义标签创建的理想注意力图的监督下，网络嵌入上下文信息，以便更好地理解场景并利用相关特征进行准确预测。此外，构建了一个特征共享模块，使任务特定的特征深度融合，并设计了一致性损失，使特征相互引导。

**[36]Unsupervised Monocular Depth Estimation in Highly Complex Environments**   
**[paper](https://arxiv.org/pdf/2107.13137.pdf)**  
本文中提出了基于单目视频的统一的基于图像传输的自适应框架，在白天场景上训练的深度模型可以适用于不同的复杂场景。我们只考虑训练编码器网络，并且共享相同的解码器。

**[37]BridgeNet: A Joint Learning Network of Depth Map Super-Resolution and Monocular Depth Estimation**   
**[paper](https://arxiv.org/pdf/2107.12541.pdf)**  
提出了一个深度图超分辨率（DSR）和单目深度估计（MDE）的联合学习网络。对于两个子网络的交互，采用差异化的引导策略，并相应地设计了两个桥(bridges)。一种是为特征编码过程设计的高频注意力桥，另一个是为深度图重建过程设计的内容引导桥。

**[38]CutDepth: Edge-aware Data Augmentation in Depth Estimation**   
**[paper](https://arxiv.org/pdf/2107.07684.pdf)**  
在CutDepth 中，部分深度在训练期间粘贴到输入图像上。 该方法在不破坏边缘特征的情况下扩展了变化数据

**[39]DEPTH ESTIMATION FROM MONOCULAR IMAGES AND SPARSE RADAR USING DEEP ORDINAL REGRESSION NETWORK**   
**[paper](https://arxiv.org/pdf/2107.07596.pdf) | [code](https://github.com/lochenchou/DORN_radar)**  
我们将稀疏雷达数据集成到单目深度估计模型中，并引入了一种新的预处理方法，以减少雷达提供的稀疏性和有限视野。

**[40]MSFNet:Multi-scale features network for monocular depth estimation**   
**[paper](https://arxiv.org/pdf/2107.06445.pdf)**  
设计了一个多尺度特征网络（MSFNet），由增强的多样化注意力（EDA）模块和上采样阶段融合（USF）模块组成。EDA模块采用spatial attention方法学习重要的空间信息，而USF模块则从多尺度特征融合的角度用高层语义信息补充低层细节信息。另外，我们设计了一个批次损失函数，为批次中较难的样本分配较大的损失因子。



