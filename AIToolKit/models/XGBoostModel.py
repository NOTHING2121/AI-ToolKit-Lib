from xgboost import XGBClassifier

class XGBoostModel:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42):
        """
        初始化 XGBoost 模型。
        :param n_estimators: 树的数量，默认为100。
        :param max_depth: 树的最大深度，默认为3。
        :param learning_rate: 学习率，默认为0.1。
        :param random_state: 随机数生成器种子用于结果的复现。
        """
        self.model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state)

    def get_model(self):
        """
        返回 XGBoost 模型实例。
        """
        return self.model

    def fit(self, X, y):
        """
        训练 XGBoost 模型。
        :param X: 训练数据集的特征。
        :param y: 训练数据集的目标变量。
        """
        return self.model.fit(X, y)

    def predict(self, X):
        """
        使用 XGBoost 模型进行预测。
        :param X: 预测数据集的特征。
        """
        return self.model.predict(X)
