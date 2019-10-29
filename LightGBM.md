Tree-Math
============
My study notes, contains Math behind all the mainstream tree-based machine learning models, covering basic decision tree models (ID3, C4.5, CART), boosted models (GBM, AdaBoost, Xgboost, LightGBM), bagging models (Random Forest).



Boosted Decision Tree
------------
**LightGBM**

> One Sentence Summary:   
Uses two cutting-edge techniques (GOSS & EFB) to significantly reduce the time & workload during finding the best feature and best split point in each base learner.

- **a. Difference between XGboost & LightGBM**  
The most important difference is that GBM needs to go through every possible features and split points to find the best feature and best split point. But LightGBM use GOSS technique to reduce the number of split points that should be gone through and use EFB to reduce the number of features that should be gone through, thereby increase training speed.     

  | XGboost   | LightGBM  |
  | :-------------: | :-------------: |
  | Use Exact Greedy Algorithm/Approximate Algorithm using Weighted Quantile Sketch to find the best split | Uses Histogram Algorithm/Gradient-based One-Side Sampling/Exclusive Feature Bundlin  |
  |  Level-wise tree growing strategy | Leaf-wise tree growing strategy  |
  |  Only feature parallel| Support feature parallel/data parallel/Voting parallel  |
  |  No direct support for categorical features| Directly support categorical features |

- **b. what is Histogram Algorithm**  
  Here Histogram Algorithm is very similar to the Approximate Algorithm using Weighted Quantile Sketch covered in the [XGboost section](./XGboost.md). Below is the pseudo code of this algorithm in [the original paper](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf).  

  ![img](./source_photo/lightgbm_histogram.jpg)  

  The Histogram Algorithm in LightGBM will try to discretize every feature in the dataset into bins. 
  
  
  ############
  
  float number into Ints type and put all the samples into K bins seperated by Int (e.g. (2,3], (3,4],(4,5]). For example, supose x1=2.1, then x1 should fall into bin (2, 3]. Each sample will fall into one of the bins.  
  
  Then we can go though all the samples to accumulate statistics (graident/hessian) for the histogram (contains K bins) for each feature. After accumulating this statistics, we just need to find the best int to seperate the samples in the node. This method will save RAM because we are storing Int instead of Float. This method also saves lots of computing time by only loop through k Int instead of n samples one by one. But of course, this will lose some accuracy since we are not finding the exact split point.  

  The other advantage of using Histogram Algorithm algorithm is that we can get use histogram subtraction technique. This means that a leaf's histograms in a binary tree = the histogram subtraction of its parent and its neighbor. So that we don't need to compute histogram for both leaves. 





- **c. what is GOSS(Gradient-based One-Side Sampling)**  
  Besides Histogram Algorithm, lightGBM also proposes GOSS method to better reduce RAM usage and save computing time.  

  The basic idea behind GOSS is quite similar to Adaboost. Recall in Adaboost, we keep adjusting the weights for each sample in each iteration. We decrease the weight for correctly identified samples and increase the weight for incorrectly identified samples. 
  




