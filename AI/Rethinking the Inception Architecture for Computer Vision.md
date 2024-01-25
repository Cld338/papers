---
Created: 2024-01-20
Last Modified: 2024-01-20
tags:
  - CV
DOI: "\rhttps://doi.org/10.48550/arXiv.1512.00567"
완료 여부: false
---
```text-progress-bar
progress:: 1/8.5
fill:🟩
transition:🟨
empty:◻️
prefix:[
suffix:]
length:10
```
## Abstract
&emsp;컨볼루션 네크워크는 다양한 분야의 과제에 대한 state-of-the-art 컴퓨터 비전 솔루션의 핵심입니다. 2014년도 부터 매우 깊은 컨볼루션 네트워크가 다양한 벤치마크에서 상당한 발전을 취하며 주를 이루었습니다. <font color="#ff0000">모델의 크기와 계산 비용의 증가는 대부분의 과제에서 즉각적인 품질 향상으로 이어지는 경향이 있지만 (훈련을 위한 라벨링 데이터가 충분히 주어지는 한에서), 계산 효율성과 적은 매개변수는 모바일 비전 및 빅데이터와 같은 다양한 활용 사례에서 여전히 중요한 요인입니다. Although increased model size and computational cost tend to translate to immediate quality gains for most tasks (as long as enough labeled data is provided for training), computational efficiency and low parameter count are still enabling factors for various use cases such as mobile vision and big-data scenarios.</font> 여기에서 우리는 적절히 인수분해된 컨볼루션과 공격적인 정규화를 통해 추가된 계산을 최대한 효율적으로 활용할 수 있도록 네트워크의 크기를 늘리는 방법을 탐색합니다. 우리는 ILSVRC 2012 classification challenge validation set을 통해 우리의 방법을 벤치마킹하였고, 이는 state-of-the-art에 비해 상당한 개선을 이루었습니다: 추론당 50억번의 곱셈 및 덧셈에 해당하는 계산 비용과 2500만개 미만의 파라미터를 사용하는 네트워크를 통해 단일 프레임 평가에서 21.2%의 top-1 error 그리고 5.6%의 top-5 error를 달성하였습니다. 4개의 모델을 앙상블한 뒤 multi-crop evaluation을 한 결과 3.5%의 top-5 error와 17.3%의 top-1 error가 나타남을 확인하였습니다.
## 1. Introduction
&emsp;2012년 ImageNet competition [16]에서 Krizhevsky et al [9] 이 입상한 이후, 그들의 네트워크인 "AlexNet"은 object detection [5], segmentation [12], human pose estimation [22], video classification [8], object tracking [23] 그리고 superresolution [3] 등 다양한 컴퓨터 비전 과제에 성공적으로 도입되어 왔습니다.

&emsp;이러한 성공은 더 나은 성능의 CNN을 찾는 데에 초점을 둔 새로운 연구 방향을 촉진시켰습니다. 2014년부터 더 깊고 넓은 네트워크들을 활용함으로써 네트워크 아키텍처의 성이 확연히 개선되었습니다. VGGNet [18] 과 GoogLeNet [20]은 2014 ILSVRC [16] classifiacation challenge에서 비슷하게 높은 성능을 내었습니다. 한 가지 흥미로운 관측 결과는 classification 성능이 다양한 응용 분야에서의 상당한 품질 향상으로 이어진다는 것입니다. 이는 깊은 컨볼루션 아키텍처에서의 구조적 개선이 높은 품질의 학습된 시각적 특징에 점점 더 의존하는 대부분의 다른 컴퓨터 비전 과제에서의 성능을 개선 시키는 데에 활용될 수 있음을 의미합니다. 또한, AlexNet이 수작업으로 만들어진 solution에 준하는 성능을 내지 못하는 것과 같은 경우에서(e.g. proposal generation in detection[4]),네트워크 품질의 개선이 새로운 컨볼루션 네트워크의 응용 분야를 탄생시켰습니다.

&emsp;VGGNet [18] 이 구조적 단순성에 강점을 가지지만, 이는 높은 비용을 초래합니다: 네트워크를 평가하는 데에 많은 계산이 요구됩니다. 반면에, GoogLeNet [20] 의 inception 아키텍처는 메모리와 계산 한도에 대한 엄격한 제한 내에서도 좋은 성능을 낼 수 있도록 설계되었습니다. 예를 들어, GoogLeNet은 500만개의 파라미터만을 사용하였으며 이는 600만개의 파라미터를 사용하였던 AlexNet에 비해 12배 감소한 수치입니다. 게다가, VGGNet은 AlexNet 보다 약 3배 더 많은 파라미터를 사용하였습니다.

&emsp;Inception의 계산 비용은 VGGNet이나 더 높은 성능의 후속 모델들 보다 훨씬 낮습니다 [6]. 이는 대규모 데이터를 합리적인 비용으로 처리해야 하는 빅데이터 분야나 모바일 비전과 같이 메모리 또는 계산 용량이 본질적으로 제한된 환경에서 Inception 네트워크를 활용 가능하도록 만들어왔습니다.
## 2. General Design Principles
## 3. Factorizing Convolutions with Large Filter Size
## 3.1. Factorization into smaller convolutions
## 3.2. Spatial Factorization into Asymmetric Convolutions
## 4. Utility of Auxiliary Classifiers
## 5. Efficient Grid Size Reduction
## 6. Inception-v2

## References
[1] M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S. Corrado, A. Davis, J. Dean, M. Devin, S. Ghemawat, I. Goodfellow, A. Harp, G. Irving, M. Isard, Y. Jia, R. Jozefowicz, L. Kaiser, M. Kudlur, J. Levenberg, D. Mane,´ R. Monga, S. Moore, D. Murray, C. Olah, M. Schuster, J. Shlens, B. Steiner, I. Sutskever, K. Talwar, P. Tucker, V. Vanhoucke, V. Vasudevan, F. Viegas, O. Vinyals, P. War- ´ den, M. Wattenberg, M. Wicke, Y. Yu, and X. Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org. 
[2] W. Chen, J. T. Wilson, S. Tyree, K. Q. Weinberger, and Y. Chen. Compressing neural networks with the hashing trick. In Proceedings of The 32nd International Conference on Machine Learning, 2015. 
[3] C. Dong, C. C. Loy, K. He, and X. Tang. Learning a deep convolutional network for image super-resolution. In Computer Vision–ECCV 2014, pages 184–199. Springer, 2014. 
[4] D. Erhan, C. Szegedy, A. Toshev, and D. Anguelov. Scalable object detection using deep neural networks. In Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on, pages 2155–2162. IEEE, 2014. 
[5] R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014. 
[6] K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. arXiv preprint arXiv:1502.01852, 2015. 
[7] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Proceedings of The 32nd International Conference on Machine Learning, pages 448–456, 2015. 
[8] A. Karpathy, G. Toderici, S. Shetty, T. Leung, R. Sukthankar, and L. Fei-Fei. Large-scale video classification with convolutional neural networks. In Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on, pages 1725–1732. IEEE, 2014. 
[9] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pages 1097–1105, 2012. 
[10] A. Lavin. Fast algorithms for convolutional neural networks. arXiv preprint arXiv:1509.09308, 2015. 
[11] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu. Deeplysupervised nets. arXiv preprint arXiv:1409.5185, 2014. 
[12] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3431–3440, 2015. 
[13] Y. Movshovitz-Attias, Q. Yu, M. C. Stumpe, V. Shet, S. Arnoud, and L. Yatziv. Ontological supervision for fine grained classification of street view storefronts. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1693–1702, 2015. 
[14] R. Pascanu, T. Mikolov, and Y. Bengio. On the difficulty of training recurrent neural networks. arXiv preprint arXiv:1211.5063, 2012. 
[15] D. C. Psichogios and L. H. Ungar. Svd-net: an algorithm that automatically selects network structure. IEEE transactions on neural networks/a publication of the IEEE Neural Networks Council, 5(3):513–515, 1993. 
[16] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, et al. Imagenet large scale visual recognition challenge. 2014. 
[17] F. Schroff, D. Kalenichenko, and J. Philbin. Facenet: A unified embedding for face recognition and clustering. arXiv preprint arXiv:1503.03832, 2015. 
[18] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014. 
[19] I. Sutskever, J. Martens, G. Dahl, and G. Hinton. On the importance of initialization and momentum in deep learning. In Proceedings of the 30th International Conference on Machine Learning (ICML-13), volume 28, pages 1139–1147. JMLR Workshop and Conference Proceedings, May 2013. 
[20] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1–9, 2015. 
[21] T. Tieleman and G. Hinton. Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning, 4, 2012. Accessed: 2015- 11-05. 
[22] A. Toshev and C. Szegedy. Deeppose: Human pose estimation via deep neural networks. In Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on, pages 1653–1660. IEEE, 2014.
[23] N. Wang and D.-Y. Yeung. Learning a deep compact image representation for visual tracking. In Advances in Neural Information Processing Systems, pages 809–817, 2013.