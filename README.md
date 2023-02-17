# resnet_cifar10

ResNet은 2015년에 나온 논문으로 ILSVRC이미지 인식 대회에서 1위한 모델. 진정한 Deep Learning의 시대를 가져왔다고 말할 정도로 영향력이 큰 네트워크이다.

23년 1월1일인 지금에도 어김없이 Resnet은 명실상부라고 할 수 있다. 그래서 아래 참고에 넣어둔 Resnet의 논문을 **1. 간단하게 리뷰**하고 실제로 **2. cuda pytorch 기반에서 Resnet을 이용하여 Cifar10을 돌려보려 한다**.

본 실험에서는 Intel(R) Core(TM) i5-6600 CPU @ 3.30GHZ,  Ram 32.0GB, NVIDIA GeForce RTX 3060 환경에서 

Python == 3.7.15   
cuda ver. == 11.3  
Pytorch == 1.12.1   
를 사용하여 실험환경을 구축하였고 

성능평가를 위해 Train과 Test data set에 대한 Accuracy와 Cross Entropy Loss를 사용하였다.


