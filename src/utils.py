import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
def evalute_model(x_train,y_train,x_test,y_test,models:dict):
    try:
        report = {}
        for name, model in models.items():
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            score = r2_score(y_test, y_pred)
            report[name] = score
        return report
    except Exception as e:
        raise CustomException(e,sys)
