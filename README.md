# [Leverage Your Local and Global Representations: A New Self-Supervised Learning Strategy](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Leverage_Your_Local_and_Global_Representations_A_New_Self-Supervised_Learning_CVPR_2022_paper.pdf)

This readme file is an outcome of the [CENG502 (Spring 2023)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 2023) Project List](https://github.com/CENG502-Projects/CENG502-Spring2023) for a complete list of all paper reproduction projects.

# 1. Introduction

The paper, namely "Leverage Your Local and Global Representations: A New Self-Supervised Learning Strategy" by Zhang et al., is published as a conference paper on CVPR 2022. It proposes a new self-supervised learning (SSL) strategy *_-LoGo-_* that can be adapted on top of existing SSL methods such as [MoCo](https://arxiv.org/abs/1911.05722)[] (denoted as _MoCo-LoGo_) and [SimSiam](https://arxiv.org/abs/2011.10566)[] (denoted as _SimSiam-LoGo_).

This implementation focuses on SimSiam-LoGo, implying that the proposed technique is applied to SimSiam. The goal of this repository is to reproduce some of the results belonging to the section 4.2 of the paper: _"Training and evaluating the features"_.

## 1.1. Paper summary

A plethora of SSL methods rely on maximizing the similarity between representations of the same instance *(some further take negatives into account by methods such as large batch-sizes [] or queues [])*. These representations are aimed to be view-invariant, and thus origin from different views of the instance by applying random augmentations including random cropping.

These approaches are inherently limited by the fact that two random crops from the same image may be dissimilar, encoding different semantics. Pushing these features -regardless of the content- to be similar creates a bottleneck on the quality of the learned representations.

This paper addresses this problem by explicitly reasoning about the **Lo**cal and **G**l**o**bal views (crops) by:

1. Pulling global representations of the image together
2. Pulling local representations towards global representations of the image
3. Repelling local representations of the same image apart

# 2. The method and my interpretation

## 2.1. The original method

The method is aimed to be general, being applicable in both contrastive and non-contrastive scenarios. So, let's first do a quick review to lay the ground for the approach.

### 1. Similarity Loss

Similarity loss functions are used as objectives to maximize the similarity between two views. Two commonly used loss functions are:

1. Info-NCE [] loss (known as the contrastive loss), used by contrastive methods such as MoCo []:

![](assets/infonce.png)

Contrastive loss further accounts for the negatives, meaning that it pushes to minimize the similarity of the representation to the negatives, where negative samples may be drawn from the batch [], or from a queue [].

2. Cosine loss, used by non-contrastive methods such as SimSiam []:

![](assets/cosine.png)

where either stop-gradient operation is applied [] on z_2 branch or momentum encoder is used to obtain the representation z_2 [].

### 2. The Method

Given the complex image content of the contemporary datasets, the driving motivations behind this method are:

1. Two different crops of the same image may capture entirely different semantics
2. Two random crops from different images may capture similar semantics, conversely

Hence, it is _suboptimal_ to consider views as positive only if they originate from the same image. To address this, LoGo proposes two different kinds of crops, i) local crops, and ii) global crops.

Specifically, each image is augmented twice, using the local and global set of augmentations, resulting in two global crops, and two local crops. Then, global-to-global, local-to-global and local-to-local relationships are optimized.

_Note that l_s denotes a similarity loss from the set above. Specifically for this implementation of SimSiam-LoGo, it denotes the Cosine Loss._

#### Global-to-global
As global views cover most of the semantics of an image, maximizing the similarity between global views of the same image, and _optionally_ (for contrastive methods) minimizing the similarity between different images is aimed. The objective used here is:

![](assets/l_gg.png)

#### Local-to-global
Global crops are much more likely to capture the overall semantics of the image, while also sharing some of the semantic signal with the local crops. Therefore, they are treated as constant by either applying stop-gradient operation (SimSiam-LoGo) or fixing their representation in the momentum encoder (both denoted by sg()). The objective, then, is:

![](assets/l_lg.png)

#### Local-to-local
Contrary to most of the existing works, local crops from the same image are **encouraged** to be dissimilar, since they most likely depict different parts of an object or even entirely different objects. 

*Here, one should note that encouraging dissimilarity at some lovel is also a necessity to prevent collapse or trivial solutions.*

The objective used to encourage the dissimilarity is:

![](assets/l_ll.png)

where l_a denotes an affinity function (the higher, the more similar).

In principle, l_a can be any well-known similarity measure, such as the cosine similarity. On the other hand, however, the high-dimensional nature of the encoded representations would allow representations to be distant -using such metrics- in many directions, most of them being _meaningless_.

To incorporate more meaning _(and possibly ease the convergence of training)_, the method proposes to jointly train and use a _learned metric_ implemented as an MLP. The objective used to train it is motivated by the intuition that local crops from the same image are, on average, expected to encode more similar semantics, compared to local crops from different images. Hence, yielding the objective to be maximized:

![](assets/omega.png)

where f() denotes the MLP, z_1 & z_2 denotes local crops originating from the same image, and z_- denotes local crops from different images.

The overall objective becomes:

![](assets/overall.png)

where lambda balances the similarity and dissimilarity terms (set to 1e-4 for SimSiam-LoGo).

## 2.2. Our interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.

## 3.2. Running the code

```
conda create --name logo python=3.9
conda activate logo
```

```
pip install -r requirements.txt
```

### ImageNet100
```
python main.py --arch resnet34 --dataset imagenet100 --batch_size 384 --num_workers 16
```


## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

Erce Güder - guder.erce@metu.edu.tr
Ateş Aytekin - atesaytekinn@gmail.com