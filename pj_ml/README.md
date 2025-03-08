# Project ML: NN vs. Traditional ML Models

## Research question and Hypothesis
RQ:
Does a Feedforward Neural Network outperform traditional ML models in predicting heart disease risk?

Hypothesis:
The Feedforward NN will outperform logistic regression, decision trees, and kNN due to its ability to model more complex relationships.

## Short breakdown of what is useful for our project
For the data, we need to use a standard scaler for the age. The rest of the data does not need encoding.

- Feedforward Neural Network (Non-Linear)
The hidden layers will use a ReLU activation function, and the results coming from the hidden layers will go through a sigmoid function altogether. Without activation function the NN would just be a linear model.
The weights and biases will be initialised randomly at first. Then, through gradient descent, the Neural Network should improve its weights and biases so that it better fits the data. Therefore, we compare the predicted output values with the true label, like what we are used to. So we basically compute the loss, and we do that with a logarithmic loss function, or simply log loss. This measures how well the model predicts probabilities for binary outcomes. With backpropagation, we will compute how much each weight contributes to the error (loss). The gradient of the loss can then be found w.r.t each weight using calculus. However, this can all be done with code eventually.

- Logistic Regression (Linear Model)
This will predict a probability between 0 and 1. We will have to decide what the decision boundary of low and high risk should be, but most likely 0.5. We will, again, use log loss as our loss function.

- Linear SVM (Linear Model)
The Linear SVM searches the optimal decision boundary. This is basically maximising the margin between 2 classes. The loss function used will be Hinge Loss. This is a different function, but this is the loss function that is applied to a SVM.

- Decision Tree (Non-Linear Model)
For a decision tree, it works slightly different compared to other models. Namely, we are working with a tree. Eventually, after going over the tree, it will lead to a decision of someone either having a low or high risk on heart disease. It depends on which leaf node it ends up on.

- kNN (Non-Linear Model)
kNN classifies based on similarity on existing data. For each input, the Euclidian distance to the surrounding nodes is calculated, and the k nearest data points determine a low or high risk. If most of the neighbours are low risk, it will be classified low risk, and vice versa. For a draw, we still need to decide what to do.

Our goal is binary classication (0/1 - Low/High risk on heart disease)

## Evaluating the models
We will evaluate all the models by drawing an ROC curve. Therefore, we first need to calculate precision, recall, TPR and FPR. Then, we can use the TPR and FPR, which form our ROC space. This results in an ROC curve. We can then turn the classifier into a ranking classifier, and rank how negative and positive a point is. This will give a coverage matrix, from which we can then estimate the AUC. The model with the largest AUC will be chosen as the best model for our research question.

## Important to do (and mention in the report)
1. Split your data into train, validation, and test data. Sample randomly.
2. Choose your model, hyperparameters, etc. only using the training set. Save your test set until the very last minute. Don't use it for anything.
3. State your hypothesis.
4. Test your hypothesis once on the test data. This is usually at the very end of the project when you write the report.

## Notes after reviewing the data and testing it on a model
The data is very trivial. Therefore, most likely all classifiers will perform very well. To still try and compare the models, we should look at specific details.

Possible things to consider:
- False negatives can be very critical in real life health care. We should therefore compare confusion matrices, and decide which is the best in this sense.
- We could introduce some noise to the input data, to see which model best handles some noise in the data.
- Compare interpretability of each model. Logistic regression is easier to interpret than a feedforward NN.
- We could also investigat ewhether one model misclassifies specific types of instances more frequently than another.

For the whole principle, check out "Experiments" at https://mlvu.github.io/evaluation/