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

  The Histogram Algorithm in LightGBM will try to discretize every feature in the dataset into sets of bins. For example, feature A will be converted in k bins and samples that have A feature value between [0,0.1) will fall into bin1, and [0.1,1) might fall into bin2 etc.  

  ![img](./source_photo/lightgbm_bins.png)  

  - How to create bins?  
    LightGBM can handle both categorical features and continuous features. 

    - For continuous features C  
      Input: n samples in that parent node.  
      
      Based on n samples, we can calculate the number of distinct values U in them, and then group samples by these distinct values. We also need to predefine the number of bins B we want and also the maximun number of samples in one bins M.  

      - If U <= B, then each bins will only contain samples of one specific unique value. 
      - If U > B, then there will be bins hosts more than one unique value groups. We then count the number of samples in each unique value groups.  
      
        For those unique value groups with sample count more than n/B, each of them will fall into a distinct bin. 
        
        For the rest of the unique value groups, we will presort them based on the value of feature C then accumulate them from small to large. When the accumulated count reach M, we finish one bins. So on and forth untill all the samples fall into bins.  

    - For Categorical feature C  
      Input: n samples in that parent node.  

      Based on n samples, we can calculate the number of distinct categories U in them, and then we will discard categories with less than 1% of total samples. Each of the rest categories will fall into a bin (one category inside one bin).
    
  - How to find the best split points?  

    - For continuous feature C  
      After building up the buckets, we will go though all the samples to accumulate statistics (a score based on graident & hessian) for the histogram (contains K bins) for each feature. After accumulating this statistics, we just need to find the best split point giving the maximum gain when seperate k bins into two parts.

    - For categorical feature C

      Before splitting, We need to predefine maximum number of features that will be applied one vs others method called max_cat_to_onehot and maximum bins that will be searched called max_cat_threshold.  

      - If U <= max_cat_to_onehot, then we will use one vs others method. For example, if there are three distinct category a1, a2, a3 in feature C. Then we will only explore [a1, a2&a3], [a2, a1&a3], [a3, a1&a2] three different possbile split and find out the split that output maximum gain.  

      - If U > max_cat_to_onehot, then we will accumulate statistics for each category. Later, we will sort these category based on the value of the (sum(gradients) / sum(hessians)). Then from large to small and also from small to large each try max_cat_threshold times possible split point to find the one gives maximum gain.  

  - What are the advantages of using Histogram Algorithm?  

    In the [official website of LightGBM](https://github.com/microsoft/LightGBM/blob/master/docs/Features.rst), it listed 4 main advantages:  
    - Reduced cost of calculating the gain for each split
      - Pre-sort-based algorithms have time complexity O(#data)  
      - Computing the histogram has time complexity O(#data), but this involves only a fast sum-up operation. Once the histogram is constructed, a histogram-based algorithm has time complexity O(#bins), and #bins is far smaller than #data.  

    - Use histogram subtraction for further speedup  
      - To get one leaf's histograms in a binary tree, use the histogram subtraction of its parent and its neighbor  
      - So it needs to construct histograms for only one leaf (with smaller #data than its neighbor). It then can get histograms of its neighbor by histogram subtraction with small cost (O(#bins))  

      ![img](./source_photo/lightgbm_histogram_sub.png) 

    - Reduce memory usage  
      - Replaces continuous values with discrete bins. If #bins is small, can use small data type, e.g. uint8_t, to store training data  
      - No need to store additional information for pre-sorting feature values  

    - Reduce communication cost for parallel learning  

- **c. what is GOSS(Gradient-based One-Side Sampling)**  
  Besides Histogram Algorithm, lightGBM also proposes GOSS method to better reduce RAM usage and save computing time.  

  The basic idea behind GOSS is quite similar to Adaboost. Recall in Adaboost, we keep adjusting the weights for each sample in each iteration. We decrease the weight for correctly identified samples and increase the weight for incorrectly identified samples.  

  