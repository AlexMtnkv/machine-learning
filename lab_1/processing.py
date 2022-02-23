import pandas as pd
import pathlib
import hashlib
import numpy as np
import random
from PIL import Image
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class DataInit:
    def __init__(self, student_id="MITENKOV_ALEKSANDR_411",
                 train_directory=pathlib.Path("D:/ML/machine-learning/data/train"),
                 sample_size=5000):

        self.student_id = student_id
        self.train_directory = train_directory
        self.sample_size = sample_size
        self.target_df = None
        self._features = None
        self._target = None

    def initialize_random_seed(self):
        """Инициализирует ГПСЧ из STUDENT_ID"""
        sha256 = hashlib.sha256()
        sha256.update(self.student_id.encode("utf-8"))

        fingerprint = int(sha256.hexdigest(), 16) % (2 ** 32)

        random.seed(fingerprint)
        np.random.seed(fingerprint)

    def read_target_variable(self):
        """Прочитаем разметку фотографий из названий файлов"""
        target_variable = {
            "filename": [],
            "is_cat": []
        }
        image_paths = list(self.train_directory.glob("*.jpg"))
        random.shuffle(image_paths)
        for image_path in image_paths[:self.sample_size]:
            filename = image_path.name
            class_name = filename.split(".")[0]
            target_variable["filename"].append(filename)
            target_variable["is_cat"].append(class_name == "cat")

        self.target_df = pd.DataFrame(data=target_variable)

    def read_data(self):
        """Читает данные изображений и строит их признаковое описание"""
        image_size = (100, 100)
        features = []
        target = []
        for i, image_name, is_cat in tqdm(self.target_df.itertuples(), total=len(self.target_df)):
            image_path = str(self.train_directory / image_name)
            image = Image.open(image_path)
            image = image.resize(image_size)  # уменьшаем изображения
            image = image.convert('LA')  # преобразуем в Ч\Б
            pixels = np.asarray(image)[:, :, 0]
            pixels = pixels.flatten()
            features.append(pixels)
            target.append(is_cat)
        self._features = np.array(features)
        self._target = np.array(target)

    def model_results(self):
        X, y = self._features, self._target
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=0.6,
                                                            test_size=0.4)

        X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test,
                                                            train_size=0.5,
                                                            test_size=0.5)
        print("Non-normalized model:")
        self.model_training(X_t=X_train, y_t=y_train,
                            X_tt=X_test, y_tt=y_test,
                            X_v=X_valid, y_v=y_valid)

        print("Normalized model: ")
        scaler = StandardScaler()
        X_train_ss = scaler.fit_transform(X_train, y_train)
        X_val_ss = scaler.fit_transform(X_valid, y_valid)
        X_test_ss = scaler.transform(X_test)
        self.model_training(X_t=X_train_ss, y_t=y_train,
                            X_tt=X_test_ss, y_tt=y_test,
                            X_v=X_val_ss, y_v=y_valid)

    @staticmethod
    def model_training(**kwargs):
        """Обучение модели"""
        X_train, y_train = kwargs["X_t"], kwargs["y_t"]
        X_test, y_test = kwargs["X_tt"], kwargs["y_tt"]
        X_valid, y_valid = kwargs["X_v"], kwargs["y_v"]

        max_score = 0
        final_clf = SGDClassifier()
        for eta in np.linspace(0.1, 6, 20):
            clf = SGDClassifier(alpha=0.01,
                                learning_rate="constant",
                                eta0=eta).fit(X_train, y_train)
            y_pred = clf.predict(X_valid)
            metric = accuracy_score(y_valid, y_pred)
            if metric > max_score:
                max_score = metric
                final_clf = clf
        y_pred = final_clf.predict(X_test)
        print(f"\tModel score: ", accuracy_score(y_test, y_pred))

    @property
    def features(self):
        return self._features

    @property
    def target(self):
        return self._target
