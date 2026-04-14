# 각 데이터셋이 불러와지지 않는 경우, 아래 코드를 이용해 업로드한 데이터 파일을 불러오세요 (.ipynb와 동일 경로가 아닌 경우 각자 경로 조정 필요)

### Iris dataset

import numpy as np
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

iris_df = pd.read_csv("/content/drive/MyDrive/dataset/iris.csv")

x = iris_df.iloc[:, :-1]
y = iris_df.iloc[:, -1]

feature_names = list(x.columns)
target_names = list(y.unique())

x = x.to_numpy()
y = y.map({"setosa": 0, "versicolor": 1, "virginica": 2}).to_numpy()

mushroom_df = pd.read_csv("/content/drive/MyDrive/dataset/mushroom.csv")

x = mushroom_df.drop(columns="class")
y = mushroom_df["class"]

wine_df = pd.read_csv("/content/drive/MyDrive/dataset/wine.csv")

x = wine_df.drop(columns="target")
y = wine_df["target"]

feature_names = list(x.columns)
target_names = ["class_0", "class_1", "class_2"]

x = x.to_numpy()
y = y.map({"class_0": 0, "class_1": 1, "class_2": 2}).to_numpy()

# 각자 필요시 바로 아래 셀을 실행해 numpy, pandas, scikit-learn, matplotlib 설치

설치된 사람도 아래 코드를 실행하면 없는 경우에 설치해줌

!pip install numpy pandas scikit-learn matplotlib

# 라이브러리 불러오기

import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.datasets import load_iris

# IRIS 데이터셋을 이용한 Gaussian Naive Bayes Classifier 실습

iris = load_iris()

print(type(iris))

x = iris.data
y = iris.target

print(type(x))

print(type(y))

print(x.shape)

print(y.shape)

feature_names = iris.feature_names
target_names = iris.target_names

print(feature_names)

print(target_names)

iris_df = pd.DataFrame(x, columns=feature_names)

iris_df.head()

iris_df["species"] = y

iris_df.head()

### Iris dataset 설명

붓꽃(Iris)의 종 분류를 위한 형태 측정값 데이터 (샘플수: 150, 변수: 4개의 연속형 변수)

데이터 속성 변수: sepal length(꽃받침 길이), sepal width(꽃받침 너비), petal length(꽃잎 길이), petal width(꽃잎 너비)

타겟 변수(3개 클래스): Setosa, Versicolor, Virginica


print(iris_df["species"].value_counts())

## 원본 데이터에 Gaussian Naive Bayes 적용 (train/test 분리 X)

gnb = GaussianNB()

gnb.fit(x, y)

print("prior probabilities for classes:", gnb.class_prior_)

print("mean for classes:\n", gnb.theta_)

print("variance for classes:\n", gnb.var_)

y_pred = gnb.predict(x)

from sklearn.metrics import accuracy_score, confusion_matrix

acc = accuracy_score(y, y_pred)
print("Accuracy:", acc)

conf_mat = confusion_matrix(y, y_pred)
print(conf_mat)

y_proba = gnb.predict_proba(x)
print(y_proba.shape)

print(np.round(y_proba, 2))

## Train/Test 분할 후, train data로 분류기 학습 후 train/test 데이터에 대해 예측값 계산 및 평가

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2 y =stratify)

gnb2 = GaussianNB()
gnb2.fit(x_train, y_train)

print("Prior probabilities:", gnb2.class_prior_)
print("Mean:\n", gnb2.theta_)
print("Variance:\n", gnb2.var_)

y_train_pred = gnb2.predict(x_train)
y_test_pred = gnb2.predict(x_test)

acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_test_pred)

print("Accuracy (Train):", acc_train)
print("Accuracy (Test):", acc_test)

conmat_train = confusion_matrix(y_train, y_train_pred)
conmat_test = confusion_matrix(y_test, y_test_pred)

print("Confusion matrix (Train):\n", conmat_train)
print("Confusion matrix (Test):\n", conmat_test)

## stratify를 안하면, 발생하는 변화

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

gnb2 = GaussianNB()
gnb2.fit(x_train, y_train)

print("Prior probabilities:", gnb2.class_prior_)

원본 데이터셋의 1/3, 1/3, 1/3 비율을 유지하지 않고, 랜덤으로 나누어지기 때문에 class_prior가 훈련/테스트 데이터에서 달라짐

# Mushroom 데이터셋을 이용한 Categorical Naive Bayes Classifier 실습

from sklearn.datasets import fetch_openml

mushroom = fetch_openml(name="mushroom", version=1, as_frame=True)

x = mushroom.data
y = mushroom.target

type(x)

type(y)

print(x.shape)

print(y.shape)

print(x.columns)

x.head()

### Mushroom dataset: 식용버섯/독버섯을 분류하기 위한 범주형 데이터셋

총 8,124개의 샘플에 대해 22개의 범주형 속성 변수 측정

타겟 변수
* e - 식용 가능한(edible) 버섯
* p - 독성 있는(poisonous) 버섯

속성 변수: cap-shape(갓의 모양), cap-surface(갓의 표면 질감) 등 버섯의 특징 분류

print(x.dtypes)

y_count = y.value_counts()
print(y_count)

# 결측값 존재 여부 확인
x.isnull().sum()

결측값이 포함된 데이터가 존재하여 적절한 전처리가 필요함.

아래 코드에서 하나의 방법을 선택해 처리.

1. 결측치가 포함된 데이터 샘플 제거 (8,124개 샘플 중 2,480개 샘플 제거)
2. 결측치가 포함된 변수가 1개 뿐이므로, 해당 변수를 제거

# Option 1. 결측치가 포함된 데이터 샘플 제거

# 1-1. x에서 dropna를 이용해 결측치가 포함된 샘플을 제거.
print(x.shape)
x_clean = x.dropna()
print(x_clean.shape)

# 1-2. x에 남은 샘플들에 대해서만 y값 추출.
print(y.shape)
y_clean = y[x_clean.index]
print(y_clean.shape)

# Option 2. 결측치가 포함된 변수 제거
# (결측치를 포함한 변수가 1개뿐이기 때문에 고려해볼 수 있는 대안임. 결측치를 포함한 변수가 많은 데이터를 다루게되면 고려 X).

# 2-1. 결측치가 포함된 변수를 drop으로 제거
print(x.shape)
x_clean = x.drop(columns=["stalk-root"])
print(x_clean.shape)

# 2-2. 샘플은 그대로 변화 없으므로, y를 그대로 사용
y_clean = y

x_clean.head()

print(x_clean["cap-shape"].unique())

CategoricalNB는 범주형 변수들이 숫자로 encoding 되어 있기를 원함

예를 들어, x["cap-shape"]의 6가지 범주는 0, 1, 2, 3, 4, 5와 같이 대응되어 숫자로 변환되어야 함.

이걸 해주는 게 sklearn.preprocessing의 OrdinalEncoder

n_categories = list(x_clean.nunique())

from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()
ord_enc.fit(x_clean)

x_clean_enc = ord_enc.transform(x_clean)
x_clean_enc

x_train, x_test, y_train, y_test = train_test_split(x_clean_enc, y_clean, test_size=0.2, stratify=y_clean)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

cnb = CategoricalNB(min_categories=n_categories)

cnb.fit(x_train, y_train)

y_train_pred = cnb.predict(x_train)
y_test_pred = cnb.predict(x_test)

acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_test_pred)

print(acc_train)
print(acc_test)

conmat_train = confusion_matrix(y_train, y_train_pred)
conmat_test = confusion_matrix(y_test, y_test_pred)

print(conmat_train)

print(conmat_test)

# Wine dataset에 대해 실습

## 요구사항
### Train/Test 분할 (테스트 20%)
### 적합한 Naive Bayes 모델 선정
### 학습 데이터에 모델 학습 후, 학습/테스트 데이터에 대한 정확도, Confusion matrix 출력

from sklearn.datasets import load_wine

wine = load_wine()

x = wine.data
y = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

