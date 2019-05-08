# Breast-Cancer-Diagnostic-Prediction
## Background of FNA 
The accuracy of visually diagnosed breast fine needle aspirates (FNA) is over 90%. The overall accuracy was 94.3% in a 37-series study. Individually, the mean sensitivity for these series was 0.91 +/- 0.07 and the mean specificity was 0.87 +/- 0.18. The relatively high standard deviations indicate that the accuracy achieved in individual series varies considerably and reflects the subjectivity of visual diagnosis.   

The subjectivity which is inherent in visual diagnosis can be minimized with computer based digital image analysis and machine learning techniques. This technology will enhance the usefulness of fine needle aspiration as a diagnostic tool for breast cancer.   
(cite from **_Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1994). Machine learning techniques to diagnose breast cancer from image-processed nuclear features of fine needle aspirates. Cancer letters, 77(2-3), 163-171._**)
<p align="center">
  <img width="300" height="200" src="https://github.com/phylliskaka/Breast-Cancer-Diagnostic-Prediction/blob/master/readmeimage/cancel.jpg">
</p>

## Dataset 
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. n the 3-dimensional space is that described in:   
[K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

Also can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

Attribute Information:

1) ID number 2) Diagnosis (M = malignant, B = benign) 3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter) b) texture (standard deviation of gray-scale values) c) perimeter d) area e) smoothness (local variation in radius lengths) f) compactness (perimeter^2 / area - 1.0) g) concavity (severity of concave portions of the contour) h) concave points (number of concave portions of the contour) i) symmetry j) fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in **_30 features_**. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits. Missing attribute values: none. Class distribution: 357 benign, 212 malignant

## Data visualization 
To understand about what is going on data, we need some basic information about data, like classes percentage, variance, standart deviation, number of samples (count) or max min values. In order to visualizate data we are going to use seaborn plots package.  

### Dataset classes 
<p align="center">
  <img width="600" height="300" src="https://github.com/phylliskaka/Breast-Cancer-Diagnostic-Prediction/blob/master/readmeimage/data_visual.png">

### Volin plot
histogram of first ten features  
<p align="center">
  <img width="600" height="600" src="https://github.com/phylliskaka/Breast-Cancer-Diagnostic-Prediction/blob/master/readmeimage/mean_dist.png">  
  
In above graph, we can understand which feature might be good for classification. For example, in texture_mean feature, median of the Malignant and Benign looks like separated so it can be good for classification. However, in fractal_dimension_mean feature, median of the Malignant and Benign does not looks like separated so it does not gives good information for classification.
  
In `Logistic_featureselect_PCA.py` file, you can draw the remaining 20 features.   

### Heat map 
To understand the correlations between all features, we want to use heatmap. 
<p align="center">
  <img width="800" height="600" src="https://github.com/phylliskaka/Breast-Cancer-Diagnostic-Prediction/blob/master/readmeimage/data_corr.png">          
  
If you want to understand two features deeper, you can use joint plot. 
 
## Feature Selection with correlation
As it can be seen in map heat figure **radius_mean, perimeter_mean and area_mean** are correlated with each other so we will use only **area_mean** cause the area_mean looks more seperable in volinplot. The same reason:

select **concavity_mean** from **compatness_mean, concavity_mean, concave points_mean**.    
select **area_se** from **radius_se, perimeter_se, area_se**.    
select **area_wors**t from **radius_worst, perimeter_worst, area_worst**.   
select **concavity_worst** from **compatness_worst, concavity_worst, concave points_worst**.    
select **concavity_se** from **compatness_se, concavity_se, concave points_se**.    
select **texture_mean** from **texture_mean, texture_worst**.    
select **area_mean** from **area_worst, area_mean**.    

After dropping 14 features, we have 16 features left. Lets do PCA to reduce data dimension! 

## Reduce dimensionality using PCA 
Import sklearn to do PCA for data. After PCA, lets keep the 95% of variance, which is first 10 principal components. 
<p align="center">
  <img width="1100" height="500" src="https://github.com/phylliskaka/Breast-Cancer-Diagnostic-Prediction/blob/master/readmeimage/component_new.png">   
  
## Logistic Regression 
### Regularization 
In this project, we want to use l1 or l2 regularization. To find the best parameters, we use GridSearchCV from sklearn. 
After comparing l1 and l2, with regularization strength of C = [0.001, 0.01, 0.1, 1, 10, 100, 1000 ], we found the best paramenter is l2(C=1).   

### Prediction result 
Accuracy: 99% on 114 testing samples.        
Confusion matrix:
<p align="center">
  <img width="500" height="400" src="https://github.com/phylliskaka/Breast-Cancer-Diagnostic-Prediction/blob/master/readmeimage/confusion_mat.png">   
