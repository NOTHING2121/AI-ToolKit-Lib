from sklearn.ensemble import RandomForestClassifier

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        初始化随机森林模型。
        :param n_estimators: 森林中的树木数量，默认为100。
        :param max_depth: 树的最大深度，如果为None，则树会在所有叶子都纯净或者小于min_samples_split的样本为止才停止生长。
        :param random_state: 随机数生成器种子用于结果的复现。
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    def get_model(self):
        """
        返回随机森林模型实例。
        """
        return self.model

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
