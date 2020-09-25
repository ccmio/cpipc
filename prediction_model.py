from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib


class PredModel:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path
        self.firstname = None

    def train(self):
        # 载入表格
        form_data = pd.read_excel(self.data_path)
        firstname = form_data.columns[0]
        # 训练集和测试集划分(8:2)
        train_set, test_set = train_test_split(form_data, test_size=0.2, random_state=42)  # 固定的随机数种子，每次划分结果一致
        train_features = train_set.drop(firstname, axis=1)
        train_labels = train_set[firstname].copy()

        # 数据预处理管道（数据清洗+归一化）
        # preprocess_pipeline = Pipeline([('imputer', SimpleImputer(missing_values=0, strategy="median")),  # 用中位数填写缺失值
        #                                 ('scaler', StandardScaler())])  # 特征放缩（归一化）
        # X = preprocess_pipeline.fit_transform(train_features)
        X = train_features

        # 随机森林
        forest_reg = RandomForestRegressor()

        # 随机搜索优化
        param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
                      {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}]
        rand_search = RandomizedSearchCV(forest_reg, param_grid, n_iter=36, cv=5, scoring='neg_mean_squared_error')
        rand_search.fit(X, train_labels)
        final_model = rand_search.best_estimator_
        # 保存模型
        joblib.dump(final_model, self.model_path)
        # 使用模型进行预测
        # 载入表格

    @staticmethod
    def predFunc(input_from_data, mode):
        # 读取数据
        firstname = input_from_data.columns[0]
        X_input = input_from_data.drop(firstname, axis=1)
        y_input = input_from_data[firstname].copy()
        imputer = SimpleImputer(missing_values=0, strategy="median")
        scaler = StandardScaler()
        imputer.fit_transform(X_input)
        scaler.fit_transform(X_input)

        # 载入模型
        if mode == 1:
            SAVE_PATH = './model/forest_reg.pkl'
        else:
            SAVE_PATH = './model/forest_reg_S.pkl'
        final_model = joblib.load(SAVE_PATH)

        # 进行预测
        predictions_input = final_model.predict(X_input)

        # 误差分析与结果打印
        # mae_input = mean_absolute_error(y_input, predictions_input)
        # print("MAE：{}".format(mae_input))
        # print(u"预测值  实际值")
        # for predict, actual in zip(predictions_input, y_input):
        #     print("{:.2f} {:.2f}".format(predict, actual))
        return predictions_input[0]
