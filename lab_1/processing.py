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

    def model_training(self):
        """Обучение модели"""
        X, y = self._features, self._target

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=0.6,
                                                            test_size=0.4)

        X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test,
                                                            train_size=0.5,
                                                            test_size=0.5)
        sc = {}
        for _ in np.arange(0.05, 1.0, 0.05):
            clf = SGDClassifier(learning_rate="constant", eta0=_).fit(X_train, y_train)
            y_pred = clf.predict(X_valid)
            sc[_] = accuracy_score(y_valid, y_pred)

        final_eta = sorted(sc, key=sc.__getitem__)[-1]
        clf = SGDClassifier(learning_rate="constant", eta0=final_eta).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Model score: ", accuracy_score(y_test, y_pred))



    @property
    def features(self):
        return self._features

    @property
    def target(self):
        return self._target
