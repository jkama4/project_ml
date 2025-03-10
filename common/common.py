# Commonly used utilities


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class kUtils:
    _data_filepath: str = "../data/heart_disease_risk_dataset_earlymed.csv"
    _loaded_data: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def get_loaded_data() -> pd.DataFrame:
        if kUtils._loaded_data.empty:
            kUtils._loaded_data = pd.read_csv(kUtils._data_filepath)

        assert not kUtils._loaded_data.empty

        return kUtils._loaded_data

    @staticmethod
    def get_split_data(
        train_frac: float = 0.5,
        test_frac: float = 0.3,
        validation_frac: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if train_frac + test_frac + validation_frac != 1:
            raise Exception("[DATA SPLITTER]: Fraction sizes need to add up to 1.")

        loaded_data: pd.DataFrame = kUtils.get_loaded_data()

        inputs: np.ndarray = loaded_data.drop(columns=["Heart_Risk"]).to_numpy()
        outputs: np.ndarray = loaded_data["Heart_Risk"].to_numpy()

        X_train, X_temp, y_train, y_temp = train_test_split(
            inputs,
            outputs,
            test_size=1 - train_frac,
            train_size=train_frac,
        )

        test_size = test_frac / (test_frac + validation_frac)
        X_test, X_val, y_test, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=test_size,
        )

        return X_train, X_test, X_val, y_train, y_test, y_val
