---
title: 最邻近算法
comments: true
---

## 背景

给定一组含有标签的记录.

| outlook  | temp. | humidity | windy | play |
|----------|-------|----------|-------|------|
| sunny    | hot   | high     | false | no   |
| sunny    | hot   | high     | true  | no   |
| overcast | hot   | high     | false | yes  |
| rainy    | mild  | high     | false | yes  |
| rainy    | cool  | normal   | false | yes  |
| rainy    | cool  | normal   | true  | no   |
| overcast | cool  | normal   | true  | yes  |
| sunny    | mild  | high     | false | no   |
| sunny    | cool  | normal   | false | yes  |
| rainy    | mild  | normal   | false | yes  |
| sunny    | mild  | normal   | true  | yes  |
| overcast | mild  | high     | true  | yes  |
| overcast | hot   | normal   | false | yes  |
| rainy    | mild  | high     | true  | no   |

- 14条记录
- 4列属性: outlook, temp, humidity, windy
- 1列标签: play

任务就是开发一个模型, 这个模型能够基于给出的新的属性值来判断play的类型. 用于开发模型的记录集被称为训练集. 为了能够衡量模型的性能, 需要一组测试数据, 这组测试数据是不参与开发模型的, 它也是带有标签的. 

## 1-最邻近算法

记住所有的训练样本. 通过[距离测量](/algorithm/preprocessing/#相似性测量)得到离新样本(不含有标签)距离最近的样本, 这个样本的标签将会是新样本的预测标签. 

这个算法的边界可以参考Voronoi图(在计算机图形学中是一大章内容).

???+ example "例子"

	| Example | a1 | a2 | a3 | Class |
	|---------|----|----|----|-------|
	| ex1     | 1  | 3  | 1  | yes   |
	| ex2     | 3  | 4  | 5  | yes   |
	| ex3     | 3  | 2  | 2  | no    |
	| ex4     | 5  | 2  | 2  | no    |

	给出一组属性(a1=2, a2=4, a3=2), 预测它的Class.

	- D(new, ex1) = sqrt((2-1)^2 + (4-3)^2 + (2-1)^2) = sqrt(3) yes
	- D(new, ex2) = sqrt((2-3)^2 + (4-4)^2 + (2-5)^2) = sqrt(10) yes
	- D(new, ex3) = sqrt((2-3)^2 + (4-2)^2 + (2-2)^2) = sqrt(5) no
	- D(new, ex4) = sqrt((2-5)^2 + (4-2)^2 + (2-2)^2) = sqrt(13) no

	可以看出在1-邻近下, 和它最近的样本是ex2, 所以它的Class是Yes.
	
### 复杂度

没有建立模型, 只是存储了训练样本. 将每个未见过的样本和训练样本进行比较, 假设有m个已知训练样本, 每个训练样本为n维, 则查找每一个未见过的训练样本的时间复杂度是O(mn). 

对于大量的数据来说, 这个算法的效率不行. 但是也可以借助一些数据结构像是KD树或者是Ball树提高算法的效率.

## k-最邻近算法 {#knn}

使用1个最近的训练样本来预测未知样本的标签的算法称为1-邻近算法, 使用k个最近的训练样本预测未知样本的标签的算法称为k-最邻近算法, 即KNN. 

k-最邻近算法对于k的值非常敏感, 经验法则为$k\leq \sqrt{训练集}$, 商业的包一般使用的k是10. 参考更多的邻近样本能够增强抗干扰能力. K最邻近算法不仅用于分类, 还能用于回归. k-最邻近算法的结果基于k个最邻近样本的标签的均值.

???+ example "例子"

	| Example | a1 | a2 | a3 | Class |
	|---------|----|----|----|-------|
	| ex1     | 1  | 3  | 1  | yes   |
	| ex2     | 3  | 4  | 5  | yes   |
	| ex3     | 3  | 2  | 2  | no    |
	| ex4     | 5  | 2  | 2  | no    |

	给出一组属性(a1=2, a2=4, a3=2), 预测它的Class.

	- D(new, ex1) = sqrt((2-1)^2 + (4-3)^2 + (2-1)^2) = sqrt(3) yes
	- D(new, ex2) = sqrt((2-3)^2 + (4-4)^2 + (2-5)^2) = sqrt(10) yes
	- D(new, ex3) = sqrt((2-3)^2 + (4-2)^2 + (2-2)^2) = sqrt(5) no
	- D(new, ex4) = sqrt((2-5)^2 + (4-2)^2 + (2-2)^2) = sqrt(13) no

	可以看出在3-邻近下, 和它最近的样本是ex1, ex2, ex3, 由于yes的数量多于no的数量, 所以它的Class是Yes.

## 加权最邻近算法

这个算法的思想是离未知样本更近的训练样本的权值应该更大. 