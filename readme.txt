Hello kind grader, code for this project can be found here: https://github.com/bjs250/CS7641

It's written to be compatible with Python Version 3.8.0, and you can install the libraries to run it in your virtual
environment from reqs.txt

Each supervised learning method has its own file (e.g. decision_tree.py) with a set of flags configured
in the __main__ method for whatever experiment you wish to run, from generating learning curves,
running an experiment on a hyperparameter, or evaluating model performance.

The datasets can be found in the "data" directory, various figures from experiments in the "figures" directory, 
and pickled parameters to load from hyperparameter tuning.

I stole a ton of code and inspiration from the following resources:

Processing & Plotting utilities
https://pbpython.com/categorical-encoding.html
https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

DT
https://www.datacamp.com/community/tutorials/decision-tree-classification-python
https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680
https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
https://www.analyticsvidhya.com/blog/2020/10/cost-complexity-pruning-decision-trees/

Boosted DT
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html

SVM
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python

kNN
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

NN
https://stackabuse.com/introduction-to-neural-networks-with-scikit-learn/
https://panjeh.medium.com/scikit-learn-hyperparameter-optimization-for-mlpclassifier-4d670413042b
https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
