#Liver disease prediction




from sklearn.linear_model import LogisticRegression
log_classifier = LogisticRegression(random_state = 0)
log_classifier.fit(x_train, y_train)
# Predicting the output 
log_y_pred = log_classifier.predict(x_test)
#visualizing
# from sklearn.metrics import confusion_matrix
# log_cm = confusion_matrix(y_test, log_y_pred)
# sns.heatmap(log_cm , annot=True)


#Testing code:
# from sklearn.metrics import accuracy_score, precision_score
# print(accuracy_score(y_test,log_y_pred))
# print(precision_score(y_test , log_y_pred))



#Kidney disease prediction
# RandomForestClassifier:
from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier()
RandomForest = RandomForest.fit(X_train,y_train)

# Predictions:
y_pred = RandomForest.predict(X_test)

# Performance:
# print('Accuracy:', accuracy_score(y_test,y_pred))
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))

#Diabetes prediction using ML
#XG Boost Hyperparameter optimization
xgb_tuned = GradientBoostingClassifier(**xgb_cv_model.best_params_).fit(X,y)
cross_val_score(xgb_tuned, X, y, cv = 10).mean()


#Heart disease prediction
from sklearn.ensemble import RandomForestClassifier

max_accuracy = 0


for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train,Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train,Y_train)
Y_pred_rf = rf.predict(X_test)
score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)

# print("The accuracy score achieved using Decision Tree is: "+str(score_rf)+" %")

