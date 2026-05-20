from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def train_model(X_train,X_test,y_train,y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=5000)
    lr.fit(X_train_scaled, y_train)
    lr_acc = lr.score(X_test_scaled, y_test)

    rf = RandomForestClassifier(n_estimators=100,random_state=42)
    rf.fit(X_train,y_train)
    rf_acc = rf.score(X_test,y_test)

    if rf_acc > lr_acc:
        return rf,rf_acc, "Random Forest"
    else:
        return lr,lr_acc,"Logistic Regression"

