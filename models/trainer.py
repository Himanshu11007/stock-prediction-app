from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def train_model(X_train,y_train,X_test,y_test):
    lr = LogisticRegression(max_iter = 1000)
    lr.fit(X_train,y_train)
    lr_acc = lr.score(X_test,y_test)

    rf = RandomForestClassifier(n_estimators=100,random_state=42)
    rf.fit(X_train,y_train)
    rf_acc = rf.score(X_test,y_test)

    if rf_acc > lr_acc:
        return rf,rf_acc, "Random Forest"
    else:
        return lr,lr_acc,"Logistic Refression"

