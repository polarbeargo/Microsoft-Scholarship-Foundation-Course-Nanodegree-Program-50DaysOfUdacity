# Microsoft Scholarship Foundation Course Nanodegree Program 50DaysOfUdacity


[image1]: ./images/score.png    
[image2]: ./images/evaluate.png  
[image3]: ./images/ApproachesToML.png
[image4]: ./images/import.png
[image5]: ./images/transformExport.png
[image6]: ./images/fixDtypeToString.png
[image7]: ./images/datasetsVersion.png

Participation in the Microsoft Scholarship Foundation course Nanodegree Program 50 days of Udacity challenge

Follow Udacity Git Commit Message Style Guide: https://udacity.github.io/git-styleguide/  :heart:

### Day 1 : 14/07/2020 
#### Polarbeargo 

* Reading Lesson 2 Introduction to Machine Learning section 20 Linear Regression.
* I very love the Text data section where I have chance to read how to process text data with normalization(Lemmatization, Tokenization) and Vectorization where 
  I got understand about:
    * Term Frequency-Inverse Document Frequency (TF-IDF) vectorization
    * Word embedding, as done with Word2vec or Global Vectors (GloVe)
* Section 8 Scaling data with Standardization and Normalization where I'm very glad that I have chance to read this section :)

### Day 2 : 15/07/2020 
#### Polarbeargo 

* Reanding Lesson 2 Introduction to Machine Learning section 20 Linear Regression grasp the following concepts to talk to:
  * "To train a linear regression model" means to learn the coefficients and bias that best fit the data. 
  * The process of finding the best model is essentially a process of finding the coefficients and bias that minimize this error. 
  * Preparing the Data:  
        * Linear assumption  
        * Remove collinearity  
        * Gaussian (normal) distribution  
        * Rescale data  
        * Remove noise    
  * Calculating the Coefficients: Choose a cost function (like RMSE) to calculate the error and then minimize that error in order to arrive at a line of best fit that models the training data and can be used to make predictions.  
* Writing Quiz "Linear Regression: Check Your Understanding".

### Day 3 : 16/07/2020 
#### Polarbeargo

* Reading Lesson 2 Introduction to Machine Learning section 23 lab Instruction and writing section 24 lab: Train a Linear Regression Model:  

Score Model      |  Evaluate Model
:-------------------------:|:-------------------------:
![][image1]                | ![][image2]

### Day 4 : 17/07/2020 
#### Polarbeargo

* Reading Lesson 2 Introduction to Machine Learning section 25 Learning function keypoints:  
  * Irreducible error in Learning function is caused by the data collection process—such as when we don't have enough data or don't have enough data features.  
  * Model error measures how much the prediction made by the model is different from the true output. The model error is generated from the model and can be reduced during the model learning process.  
  
* Reading section 26 parametric vs. Non-parametric keypoints:   
  * Parametric Machine Learning Algorithms: 
     * Making assumption about the mapping function and have a fixed number of parameters. 
     * No matter how much data is used to learn the model, this will not change how many parameters the algorithm has. 
     * With a parametric algorithm, we are selecting the form of the function and then learning its coefficients using the training data.
     
     Benefits:

     * Simpler and easier to understand; easier to interpret the results.
     * Faster when talking about learning from data.
     * Less training data required to learn the mapping function, working well even if the fit to data is not perfect.   
     
     Limitations:

      * Highly constrained to the specified form of the simplified function.
      * Limited complexity of the problems they are suitable for.
      * Poor fit in practice, unlikely to match the underlying mapping function.
  
  * None-Parametric Machine Learning Algorithms:  
      * Non-parametric algorithms do not make assumptions regarding the form of the mapping function between input data and output so they are free to learn any functional form from the training data such as KNN and Decision tree.  
      
      Benefits:

      * High flexibility, in the sense that they are capable of fitting a large number of functional forms.
      * Power by making weak or no assumptions on the underlying function.
      * High performance in the prediction models that are produced.  
      
    Limitations:

      * More training data is required to estimate the mapping function.
      * Slower to train, generally having far more parameters to train.
      * Overfitting the training data is a risk; overfitting makes it harder to explain the resulting predictions.
  
### Day 5 : 18/07/2020 
#### Polarbeargo  

* Reading Lesson 2 Introduction to Machine Learning section 27 Classical ML vs. Deep Learning and section 28 Approaches to Machine Learning.
* Grasp concepts:
![][image3]

### Day 6 : 19/07/2020 
#### Polarbeargo 

* Reading Lesson 2 Introduction to Machine Learning section Bias vs. Variance Trade-off.
* Keypoint concepts:
   * Bias : 
      * Measures how inaccurate the model prediction is in comparison with the true output. 
      * Due to erroneous assumptions made in the machine learning process to simplify the model and make the target function easier to learn. 
      * High model complexity tends to have a low bias.
      
   * Variance: 
      * Measures how much the target function will change if different training data is used. 
      * Variance can be caused by modeling the random noise in the training data. 
      * High model complexity tends to have a high variance. 
      
* As a general trend, parametric and linear algorithms often have high bias and low variance, whereas non-parametric and non-linear algorithms often have low bias and high variance.  
  * Low bias:  
     * Low bias means fewer assumptions about the target function.  
     * Examples of algorithms with low bias are KNN and decision trees.   
     * Having fewer assumptions can help generalize relevant relations between features and target outputs. In contrast, high bias means more assumptions about the target function. Linear regression would be a good example (e.g., it assumes a linear relationship).   
     * Having more assumptions can potentially miss important relations between features and outputs and cause underfitting.  
  * Low variance:    
     * indicates changes in training data would result in similar target functions. For example, linear regression usually has a low variance.   
     * High variance indicates changes in training data would result in very different target functions. For example, support vector machines usually have a high variance.   
     * High variance suggests that the algorithm learns the random noise instead of the output and causes overfitting.      
* Increasing model complexity would decrease bias error since the model has more capacity to learn from the training data. But the variance error would increase if the model complexity increases, as the model may begin to learn from noise in the training data.

* The goal of training machine learning models is to achieve low bias and low variance.

### Day 7 : 20/07/2020 
#### Polarbeargo  

* Writing lesson 3 Model Training Lab: Import, Transform, and Export Data:  

Import Data      |  Fix Dtype to String |  Transform, and Export Data
:-------------------------:|:-------------------------:|:-------------------------:
![][image4]                | ![][image6]               |![][image5]

### Day 8 : 21/07/2020 
#### Polarbeargo  
* Reading lesson 3 Model Training section 6 Managing Data and section 8 More about datasets.  
* Organize keypoint:  
  * Datastores offer a layer of abstraction over the supported Azure storage services. They store all the information needed to connect to a particular storage service. Datastores provide an access mechanism that is independent of the computer resource that is used to drive a machine learning process.  

  * Datasets are resources for exploring, transforming, and managing data in Azure ML. A dataset is essentially a reference that points to the data in storage. It is used to get specific data files in the datastores.  
  
* Steps of the data access workflow are:  

  1. Create a datastore so that you can access storage services in Azure.  
  2. Create a dataset, which you will subsequently use for model training in your machine learning experiment.  
  3. Create a dataset monitor to detect issues in the data, such as data drift.  
* We do versioning most typically when:  

  * New data is available for retraining.  
  * When you are applying different approaches to data preparation or feature engineering.

### Day 9 : 22/07/2020 
#### Polarbeargo  
* Writing and reading lesson 3 Model Training Lab: Create and Version a Dataset, section 11 Introducing Features and section 12 Feature Engineering.    
![][image7]  
* The columns in a table can be referred to as features, selecting the features process is called feature selection and dimensionality reduction is to decrease the number of features.  
* Feature Engineering methods:  

  * Flagging: Deriving a boolean (0/1 or True/False) value for each entity.
  * Aggregation: Getting a count, sum, average, mean, or median from a group of entities.
  * Part-of: Extracting the month from a date variable.
  * Binning: Grouping customers by age and then calculating average purchases within each group.
