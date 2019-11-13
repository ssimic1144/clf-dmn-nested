from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

class InternalClassifier():
    clf = None
    columns = None

class LayeredClassifier(BaseEstimator, ClassifierMixin):  

    def __init__(self, num_levels=5, min_purity=0.7, min_feature_importance=0.4, min_leaf_samples=0.1):
        self.num_levels = num_levels
        self.min_purity = min_purity
        self.min_feature_importance = min_feature_importance
        self.ensemble = []
        self.min_leaf_samples = min_leaf_samples

    def fit(self, X, y):
        X_ = X.copy()
        y_ = y.copy()
        self._msf = max(1, int(self.min_leaf_samples*X.shape[0]))

        for i in range(self.num_levels):
            X_, y_, c = self._extract_pure(X_, y_)
            if X_.shape[0] == 0:
                break
            self.ensemble.append(c)
            # print(f"New length: {X_.shape[0]}")
        # print("Num levels =", len(self.ensemble), "unclassified = ", X_.shape[0])

    def predict(self, X):
        prediction = pd.Series([None]*X.shape[0])

        for i, e in enumerate(self.ensemble):
            fit = e.clf
            columns = e.columns[0].values
            leaves = pd.Series(fit.apply(X[columns]))
            purity = leaves.apply(
                lambda x: max(fit.tree_.value[x][0]) / sum(fit.tree_.value[x][0]))
            select = (purity > self.min_purity).values

            p = fit.predict(X[columns])
            if i > 0:
                prediction[select & prediction.isna()] = fit.predict(X[columns])
            else:
                prediction = pd.Series(p)

        return prediction.values

    def _extract_pure(self, X, y):

        # print("Selecting important features...")
        clf = GradientBoostingClassifier(n_estimators=200, random_state=1)
        clf.fit(X, y)
        feature_importances = pd.DataFrame(zip(X.columns, clf.feature_importances_)).sort_values(by=1, ascending=False)
        final_columns = feature_importances[feature_importances[1] > self.min_feature_importance*np.mean(feature_importances[1])]
        # print("DONE! \n", final_columns)

        Xoriginal = X.copy()
        X = X[final_columns[0]]

        clf = tree.DecisionTreeClassifier(min_samples_leaf=self._msf, random_state=1)
        fit = clf.fit(X, y)
        # tree.plot_tree(fit)
        # plt.show()

        _leave = pd.Series(fit.apply(X))
        _purity = _leave.apply(lambda x: max(fit.tree_.value[x][0]) / sum(fit.tree_.value[x][0]))

        selected = (_purity < self.min_purity).values
        newX = X[selected]
        newY = y[selected]

        c = InternalClassifier()
        c.clf = fit
        c.columns = final_columns
        return newX.copy(), newY.copy(), c


if __name__ == "__main__":

    data = pd.read_csv("creditData.csv", index_col=0)
    data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    dataset = data.copy()

    ord = preprocessing.OrdinalEncoder(categories=[["none", "little", "moderate", "rich", "quite rich"]], dtype=int)
    risk = preprocessing.OrdinalEncoder(categories=[["bad", "good"]], dtype=int)

    data.saving_accounts = data.saving_accounts.fillna("none")
    data.checking_account = data.checking_account.fillna("none")
    dataset.risk = risk.fit_transform(data.risk.values.reshape(-1, 1))
    dataset.saving_accounts = ord.fit_transform(data.saving_accounts.values.reshape(-1, 1))
    dataset.checking_account = ord.fit_transform(data.checking_account.values.reshape(-1, 1))

    one = preprocessing.LabelEncoder()
    dataset.sex = one.fit_transform(data.sex.values.reshape(-1, 1))
    dataset.housing = one.fit_transform(data.housing.values.reshape(-1, 1))

    df_purpose = pd.get_dummies(data.purpose, drop_first=True)
    dataset.drop(columns=["purpose"], inplace=True)

    dataset = pd.concat([dataset, df_purpose], axis=1)
    Xoriginal = dataset.drop(columns="risk")
    yoriginal = dataset.risk
    train, test = train_test_split(dataset, test_size=0.1, random_state=1)

    X = train.drop(columns="risk")
    y = train.risk
    clf = LayeredClassifier(num_levels=5)
    clf.fit(X, y)

    Xtest = test.drop(columns="risk")
    ytest = test.risk
    pred = clf.predict(Xtest)

    clf2 = LayeredClassifier(num_levels=5, min_purity=0.65, min_feature_importance=0.01, min_leaf_samples=0.01)
    
    # f1_score(pred, ytest)
    classifiers = [clf, clf2, GradientBoostingClassifier(n_estimators=300), tree.DecisionTreeClassifier()]
    
    for c in classifiers:
        scores = cross_val_score(c, Xoriginal, yoriginal, cv=10, scoring="f1")
        print(c, "\n", np.mean(scores), np.std(scores))
