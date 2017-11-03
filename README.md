# DDT

This is the unoffical website for the implementation of DDT (Deep Descriptor Transforming) and DDT+. 

Package Official Website: http://lamda.nju.edu.cn/code_DDT.ashx

This package is developed by Mr. Chen-Lin Zhang (http://lamda.nju.edu.cn/zhangcl/) and Mr. Xiu-Shen Wei (http://lamda.nju.edu.cn/weixs/). If you have any problem about 
the code, please feel free to contact Mr. Chen-Lin Zhang (zhangcl@lamda.nju.edu.cn). 
The package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Jianxin Wu (wujx2001@gmail.com).

If you find our package is useful to your research. Please cite our paper:

Reference: 
           
[1] X.-S. Wei, C.-L. Zhang, J. Wu, C. Shen and Z.-H. Zhou. Unsupervised Object Discovery and Co-Localization by Deep Descriptor Transforming, arXiv. (https://arxiv.org/abs/1707.06397)

[2] X.-S. Wei, C.-L. Zhang, Y. Li, C.-W. Xie, J. Wu, C. Shen and Z.-H. Zhou. Deep Descriptor Transforming for Image Co-Localization. In: Proceedings of International Joint Conference on Artificial Intelligence (IJCAI’17), Melbourne, Australia, 2017.(https://arxiv.org/abs/1705.02758)

## Requirements
The code needs Matconvnet(http://www.vlfeat.org/matconvnet/).

A GPU is optional for faster speed.

## Demo
We include a subset of Object Discovery dataset(http://people.csail.mit.edu/mrub/ObjectDiscovery/) to provide an example.

First, you should change the location to your Matconvnet's location. Then change other options like gpu and pre-trained model's location.

Then you will get the results of DDT and DDT+, which include highlight regions, bounding boxes.
