from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


class Part4Classifier:

    def __init__(self):
        # Set Hyper Params here
        self.model = None
        self.gbc_params = {
            "n_estimators": 80,
            "max_depth": 4,
            "random_state": 0,
            "verbose": 1
        }
        self.rf_params = {
            "n_estimators": 500,
            "max_depth": 4,
            "random_state": 0,
            "verbose": 1
        }

    def predict(self, X):
        print("Predicting... ", len(X))
        return self.model.predict(X)

    def train(self, X, Y):
        self.train_gbc(X, Y)
        # self.save_model()

    def train_random_forest(self, X, Y):
        print("Classifier: RandomForestClassifier.\nTraining... ", len(X))
        model = RandomForestClassifier()
        model.set_params(**self.rf_params)
        model.fit(X, Y)
        self.model = model

    def train_gbc(self, X, Y):
        print("Classifier: GradientBoostingClassifier.\nTraining... ", len(X))
        model = GradientBoostingClassifier()
        model.set_params(**self.gbc_params)
        model.fit(X, Y)
        self.model = model

    def train_svc(self, X, Y):
        print("Classifier: SVC.\nTraining... ", len(X))
        self.model = SVC(verbose=True)
        self.model.fit(X, Y)

    def save_model(self, filename="Part4Classifier.pkl"):
        print("Saving Model... ", filename)
        joblib.dump(self.model, filename)

    def load_model(self, filename="Part4Classifier.pkl"):
        print("Loading Model... ", filename)
        self.model = joblib.load(filename)
