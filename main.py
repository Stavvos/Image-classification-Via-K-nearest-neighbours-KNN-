from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report


#load the data
dataset = fetch_openml("mnist_784")


#data exploration
print("\n\nThe shape of the image dataset before the split is:\n", dataset.data.shape)
print("\n\nThe shape of the labels dataset before the split is:\n", dataset.target.shape)


#split the data into training and testing, 80/20 train/test
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = 0.2)

print("\n\nthe shape of the training data after the split is: ", x_train.shape, " and the train labels: ", y_train.shape, "\n")

print("\n\nthe shape of the testing data after the split is: ", x_test.shape, " and the train labels: ", y_test.shape, "\n")



#initialise the model
print("\n\nInitialising the KNN model by commiting all the training images and labels to memory\n")
model = KNeighborsClassifier(n_neighbors = 3).fit(x_train, y_train)

#make some predictions
predictions = model.predict(x_test)

#make a confusion matrix
confusionMatrix = confusion_matrix(y_test, predictions)

#perform cross validation on the model
crossValScores = cross_val_score(model, dataset.data, dataset.target, cv = 5)

#print the results of the cross validation
print("\nThe cross validation scores are:", crossValScores)
print("The mean accuracy of the cross validation scores is:", crossValScores.mean(), " with a standard deviation of: ", crossValScores.std())

#make a classification report and print it to the console
report = classification_report(y_test, predictions, target_names = [str(i) for i in range(10)])
print("\n\nThe classification report for K-nearest neighbours model is:\n", report)


#make a simple performance report
print("\nA performance report for the KNN model classifying hand written digits:")
print("Accuracy:",accuracy_score(y_test, predictions))
print("Precision:",precision_score(y_test, predictions, average = 'weighted'))
print("Recall:",recall_score(y_test, predictions, average = 'weighted'))
print("F1-score:",f1_score(y_test, predictions, average = 'weighted'))
print("\nconfusion matrix:\n", confusionMatrix)




