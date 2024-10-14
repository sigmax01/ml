---
title: ResNet
comments: true
---

在深度学习中, 网络的"深度", 即层数通常和模型的能力成正比. 然而, 随着网络深度的增加, 一些问题也随之出现, 最突出的就是[梯度消失和爆炸问题](/algorithm/neural-network/#vanishing-gradient), 这使得深层网络难以训练.

深度残差神经网络, Deep Residual Network, 简称ResNet, 它由微软研究院何凯明等人在2015年首次提出, 在深度学习领域产生了深远的影响. 它通过一种创新的"残差学习"机制, 解决了传统深度神经网络中的梯度消失问题, 从而实现了对非常深度网络的有效训练.

ResNet的核心思想是引入残差块, Residual Block, 通过捷径连接, Shortcut Connection让信息直接跳过一层或者多层网络. 
