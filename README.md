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

# Sources/Articles
SVM and NN, Google Scholar:
https://d1wqtxts1xzle7.cloudfront.net/92951952/41_26218_v26i1_Apr22-libre.pdf?1666599482=&response-content-disposition=inline%3B+filename%3DPrediction_of_heart_diseases_utilising_s.pdf&Expires=1742578158&Signature=DkuJEZ~qWKi3scQ2aRqpFU2PbWozT4tysML8dnmf0Qd5PXDfRj99nKmaSbuNaxNPWNIHlw8RCKEdOQ7W5OjL75-DpAjH8EaaPmXeM~ZY79BDYH208aH3prP1WLbm5FSLepfhJ6zY87-cqfo97e~~ZJ-E1SRytMo-sguXMRTtmXNWrnVYRqMM6LEitcdfjM4eTg~G4U3GFDFuSVwqC7tKMJVlNcmw3NMQXCVPCPCTPo2AZXUZ7wNSlJwtXrR3JtBowJgV-g-5fBboUuK8jLfKq24WJPay3ZQV9GFqedoJIDCopuT~uo2lmS0OEZQ3wTD~hYTqhn12l5ZpRKlWW~laIw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA

Logisitic Regression:
https://pdf.sciencedirectassets.com/776627/1-s2.0-S2666285X22X00022/1-s2.0-S2666285X22000449/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFAaCXVzLWVhc3QtMSJIMEYCIQCbfTcrmd8PpUiPJoQRERwcRSv6nLISyFCagBFklTJtbQIhAPdLs4K7TWnEaFJRixodpfPzRnYNCVGQEA0schS5MojvKrwFCKn%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1IgzuwiKa7vomVrJHwToqkAUtaxXQx%2BegazNNZOjXnmRSbauwusAx5bMCKxewSkXsuCj8M4dIJFkfejkHyZdcQAMFB9xKbJPzFlOYZWtOnoixeh2zVIBXMns8HFbxMekWZMzp8PUfohOjrrx1RJQihyTQkyqwXHLf%2B2IvHrME0ld8dsJqEGU90XXQnAQm%2FNL%2FxiV39VGSXGoQRARtAPNyTCKz%2FqOniCHzZmC6VSNsxOhiF2OeNeZ9LBA9Rx140r1z3nFChKp336a28Ahm42FOJDlklyV9SAs9sewGH9Sdwi4o3ZJgZfknBETawk9APIB9AhdKStmc0mR4nRq25la9KI8QxHiH1JUiWcQlFC1aiRwAPwYJOJv0h35HBoSuQV83Zf8BAWjYH0d%2BPnEBgnqveJp8AzDzru%2FHI9expNqmbZNYeW9JWQTvzx%2B7v04nrXEzoUYb5HLx%2BMav7WlaZscwAhCtIP%2FvhAl2GlYUvbxOMBxJo2pTFo84Mjj5X1%2BdWU%2BcdDLTAaDKsMcUNVhEpN851%2FxdjPBuzRAq7qgsEoM9IV4yUdBPHvLzDPuPyfjG8NzhNzboGOPjvEtRuVsvvHv2xfaVWweBBPaFzX6nuBinnv%2FL14d5rRg5imVDV8Cxt27SShSelQdoWymqKN4m3xrZdXdL2Um78Yx0u1ol4SLr1cCemaxYyC1Nj79APKgK2djGv4H%2FxRFTEPk3B%2FbQxGsLmhLoRNFaTl2epdIHrbaasDtZmiggsQywCVY0Z54FI4dInPc2ATOkzdgqH52y2HNjf6Yv2TgSPtmluQAMfl%2FRMqvXfAEAoP7u%2BSn%2FJik8hFDUXhQE%2FK5kE43Lk8t%2BpBN1tpOZcDAxt%2Fa0SXViAykCyX3IkBSSi%2Fe3eSUCxGTLca4LNDDck%2Fa%2BBjqwAeW1JtYDF%2BJJNCWbHbcbYrWWiKfksnpdvW5901aIJJPxaMt9FP%2F7pAwuZxYR4NK4l791Q0J8nwB0F5jA36TnqHyHztmRh5wYHMC5aTlNsoE0%2Fr3mh9LwPgYJD8XJNZQrOudkLkU4IrwNant32pW%2B6lbNlkMGIv7rOh1brJuIPg6gbucPhBT3iq11f8ijDnVy1lbMUwaJDAlwGGfuNdo1PCZqe1n%2FY2j6r%2BfEHnaIc5x5&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250321T163121Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY7A452G52%2F20250321%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=a1e02a22daaabd905d9f8e83741f9ce267a957c84037548b1d9c7144688817f2&hash=f6ae18d831761160aba16a0fa9a90bc7696a19310380972de426e84f52550342&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S2666285X22000449&tid=spdf-f5814411-f875-4c5a-b321-335799382f8b&sid=f400145161c527434e6a47e92d193dbd3a6cgxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=140a575153050657070352&rr=923ed6ce6a44339e&cc=nl

Multiple ML Models Google Scholar:
https://jpmm.um.edu.my/index.php/MJCS/article/view/35980/14415