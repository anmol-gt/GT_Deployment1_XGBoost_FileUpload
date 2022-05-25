import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor
import pickle
# from sklearn.metrics import make_scorer
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Master Data 3(for model building).csv")
df.head()

df.rename(columns={'JobLevel_SSC Classes':'JobLevel_SSC_Classes',
                   'JobLevel_Senr Associate':'JobLevel_Senr_Associate',
                  'JobLevel_Temp/Unknown':'JobLevel_TempUnknown'}, inplace=True)

X = df.drop("Hours", axis=1)
y = df["Hours"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=1)

print(X_train.columns)

model1 = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
             gamma=0, gpu_id=-1, importance_type=None,
             interaction_constraints='', learning_rate=0.01, max_delta_step=0,
             max_depth=8, min_child_weight=1,
             monotone_constraints='()', n_estimators=400, n_jobs=8,
             num_parallel_tree=1, predictor='auto', random_state=1, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)

model1.fit(X_train, y_train)

pickle.dump(model1, open('XGB_pickle.pkl','wb'))
