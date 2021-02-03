

# ML Algorithms for Enteprises

Machine leaning is a field of research that formally focuses on the theory, performance, and properties of learning systems and algorithms. It is a highly interdisciplinary field building upon ideas from many different kinds of fields such as artificial intelligence, optimization theory, information theory, statistics, cognitive science, optimal control, and many other disciplines of science, engineering, and mathematics.

It has been used on a variety of problems, including recommendation engines, recognition systems, informatics and data mining, and autonomous control systems.


|Learning types 		| 			Data processing tasks 		|	 	Distinction norm 		| 	Learning algorithms 				|
|---|---|---|---|
|Supervised learning 	|	Classification/Regression/Estimation|	 Computational classifiers 	|	Support vector machine 	|
|						|										|	Statistical classifiers 	|	Naïve Bayes 			|
|						|										|								|	Hidden Markov model 	|
|						|										|								|	Bayesian networks 		|
|						|										|	Connectionist classifiers 	|	Neural networks 		|
|Unsupervised learning 	|	Clustering/Prediction 				|	Parametric 					|	K-means 				|
|						|										|								|	Gaussian mixture model 	|
|						|										|	Nonparametric 				|	Dirichlet process mixture model|
|						|										|								|	X-means 				|
|Reinforcement learning |	Decision-making 					|	Model-free 					|	Q-learning 				|
|						|										|								|	R-learning 				|
|						|										|								|	Model-based TD learning |
|						|										|								|	Sarsa learning 			|


## A. GROUP BY LEARNING STYLE
1. *Supervised learning* --- Input data or training data has a pre-determined label e.g. True/False, Positive/Negative, Spam/Not Spam etc. A function or a classifier is built and trained to predict the label of test data. The classifier is properly tuned (parameter values are adjusted)to achieve a suitable level of accuracy.

2. *Unsupervised learning* --- Input data or training data is not labelled. A classifier is designed by deducing existing patterns or cluster in the training datasets.

3. *Semi-supervised learning* --- Training dataset contains both labeled and unlabelled data. The classifier is train to learn the patterns to classify and label the data as well as to predict.

4. *Reinforcement learning* --- The algorithm is trained to map action to situation so that the reward or feedback signal is maximised. The classifier is not programmed directlyto choose the action, but instead trained to find the most rewarding actions by trial and error.

5. *Transduction* --- Though it shares similar traits with supervise learning, but it does not develop a explicit
classifier. It attempts to predict the output based on training data, training label, and testdata.

6. *Learning to learn* --- The classifier is trained to learn from the bias it induced during previous stages.

7. It is necessary and efficient to organise the ML algorithms with respect to learning methods when one need to
consider the significance of the training data and choose the classification rule that provide the greater level of accuracy.

## B. ALGORITHMS GROUPED BY SIMILARITY

### Regression Algorithms
Regression analysis is part of predictive analytics and exploits the co-relation between dependent (target) and
independent variables. The notable regression models are:
- Linear Regression, 
- Logistic Regression, 
- Stepwise Regression, 
- Ordinary Least Squares Regression (OLSR), 
- Multivariate Adaptive Regression Splines (MARS), 
- Locally Estimated Scatterplot Smoothing (LOESS), etc.

#### PRACTICAL CASE
		Estimar cantidad de alarmas de acuerdo a cantidad de transacciones en cajeros, y otras variables de cajero

### Instance-based Algorithms
Instance-based or memory-based learning model stores instances of training data instead of developing a precise definition of target function. Whenever a new problem or example is encountered, it is examined in accordance with the stored instances in order to determine or predict the target function value. It can simply replace a stored instance by a new one if that is a better fit than the former. Due to this, they are also known as winner-take-all method. Examples:
- K-Nearest Neighbour (KNN), 
- Learning Vector Quantisation (LVQ), 
- Self-Organising Map (SOM), 
- Locally Weighted Learning (LWL) etc.

#### PRACTICAL CASE
		Determinar el status de un cajero de acuerdo a evaluaciones anteriores, y segun varios factores.

### Decision Tree Algorithms
A decision tree constructs a tree like structure involving of possible solutions to a problem based on certain constraints. It is so named for it begins with a single simple decision or root, which then forks off into a number of branches until a decision or prediction is made, forming a tree.
They are favoured for its ability to formalise the problem in hand process that in turn helps identifying potential solutions faster and more accurately than others. Examples:
- Classification and Regression Tree (CART)
- Iterative Dichotomiser 3 (ID3)
- C4.5 and C5.0
- Chi-squared AutomaticInteraction Detection (CHAID)
- Decision Stump
- M5
- Conditional Decision Trees etc.


#### PRACTICAL CASE
		Decidir cuando es necesario dar mantenimiento a un cajero.
		Decidir si un cliente esta en peligro de dejar algun servicio de acuerdo a su comportamiento y el de ex-clientes.


#### Support Vector Machine (SVM)
SVM is so popular a ML technique that it can be a group of its own. It uses a separating hyperplane or a decision plane to demarcate decision boundaries among a set of data points classified with different labels. It is a strictly supervised classification algorithm. In other words, the algorithm develops an optimal hyperplane utilizing input data or training data and this decision plane in turns categories new examples. Based on the kernel in use, SVM can perform both linear and nonlinear classification.

#### PRACTICAL CASE
	Tambien clasificación supervisada.


### Clustering Algorithms
Clustering is concerned with using ingrained pattern in datasets to classify and label the data accordingly.Examples:
- K-Means,
- K-Medians
- Affinity Propagation
- Spectral Clustering
- Ward hierarchical clustering
- Agglomerative clustering
- DBSCAN
- Gaussian Mixtures
- Birch
- Mean Shift
- Expectation Maximization (EM) etc.

#### PRACTICAL CASE
		Determinar agrupaciones de objetos que no sean claras, como cajeros o clientes



### Dimensionality Reduction Algorithms
Dimensionality reduction is typically employed to reduce a larger data set to its most discriminative components to contain relevant information and describe it with fewer features. This gives a proper visualization for data with numerous features of high dimensionality and helps in implementing supervised classification more efficiently. Examples: 

- Principal Component Analysis (PCA)
- Principal Component Regression (PCR)
- Partial Least Squares Regression (PLSR)
- Sammon Mapping
- Multidimensional Scaling (MDS)
- Projection Pursuit
- Linear Discriminant Analysis (LDA)
- Mixture Discriminant Analysis (MDA)
- Quadratic Discriminant Analysis (QDA)
- Flexible Discriminant Analysis (FDA), etc.

#### PRACTICAL CASE
		Visualizar datos de altas dimensiones
		Preprocesamiento para algoritmos de clasificación

### Ensemble Algorithms
The main purpose of an ensemble method is to integrate the projections of several weaker estimators that are singly trained in order to boost up or enhance generalisability or robustness over a single estimator. The types of learners and the means to incorporate them is carefully chosen as to maximize the accuracy. Examples:
- Boosting
- Bootstrapped Aggregation (Bagging)
- AdaBoost
- Stacked Generalization (blending)
- Gradient Boosting Machines (GBM)
- Gradient Boosted Regression Trees (GBRT)
- Random Forest
- Extremely Randomised Trees etc.

#### PRACTICAL CASE
		Mejora de otros algoritmos, en entrenamiento y accuracy


## Other possibilities
### Predictive Maintenance
Manufacturing firms regularly follow preventive and corrective maintenance practices, which are often expensive and inefficient. However, with the advent of ML, companies in this sector can make use of ML to discover meaningful insights and patterns hidden in their factory data. This is known as predictive maintenance and it helps in reducing the risks associated with unexpected failures and eliminates unnecessary expenses. ML architecture can be built using historical data, workflow visualization tool, flexible analysis environment, and the feedback loop.

### Associations
The software finds associations between two actions and can assign a probability based on how often those actions occur. Example: the software may find out that a customer that buys a book by some author X also typically buys a book by some author Y. So the people who buy X and NOT Y are potential customers for Y and we may find an association rule such as 70% of people who buy X also buy Y.


### Classification
In this category, machine learning systems fit a model to some already available data in order to make predictions. Example: to classify customers as low-risk or high-risk clients, we may gather all the information we have on them and set a particular rule to tell whether or not a customer falls into one of the classes. Then new customers will be labelled as low-risk or high-risk, based on this past data.

> Written with [StackEdit](https://stackedit.io/).
