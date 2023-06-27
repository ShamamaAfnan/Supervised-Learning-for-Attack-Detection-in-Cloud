# Supervised-Learning-for-Attack-Detection-in-Cloud

### Goal
Apply Supervised Machine Learning algorithms to detect attacks in a cloud network by classifying  instances as 'Normal' or 'Attack'. 
### Dataset Overview
The dataset has 9594 instances and 38 columns. The first 4 columns are meta-data such as epoch timestamp, VM ID, domain name, and unique identifier of the domain. The next columns are Network, Memory and Disk features. The last column is the target variable which is the status of attack or normal.
### Exploratory Data Analysis
The dataset is not balanced. In the dataset, 24%  status are marked as Attack and 76% status are Normal. 
The feature correlation heatmap is ploted, the status is highly correlated to the rate of transmitted packets from the network. The status is positively correlated to the rate of transmitted bytes from the network, the rate of received packets from the network, the rate of received bytes from the network, the rate of time spent in kernel space, and the rate of time spent in userspace. The status is negatively correlated with the rate of the number of reading bytes on the vda block device, the rate of the number of write requests on the vda block device, and the rate of time spent by vCPU threads executing guest code.
### Data Preprocessing
Data Cleaning, Normalization, minority class oversampling is performed. The train test split is 70:30.
### Model Implementation
Support Vector Machine, Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbor, XG-Boost, and Naive bayes models are implemented.
### Evaluation Metric
For all the models Accuray, Precision, Recall, F1-score, Kappa Statistic are measured. Confusion matrix and ROC-Curve are ploted.
##### All the codes are included in the jupiter notebook.


