# 라이브러리 불러오기

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Breast Cancer (유방암) 예시 데이터셋에 적용

### 유방암 데이터셋 불러오기 및 x, y에 저장

breast = load_breast_cancer()

type(breast)

x = breast.data
y = breast.target

feature_names = breast.feature_names
target_names = breast.target_names

### 데이터셋 불러오기가 안되는 경우, e-class에 첨부된 파일을 다운로드받아 아래 코드를 실행하시오. (실행시 # 제거)

# df = pd.read_csv("./breast_cancer.csv")
# x = df.drop(columns="class")
# y = df["class"]

# feature_names = list(x.columns)
# target_names = list(y.unique())

# x = x.to_numpy()
# y = y.map({label: i for i, label in enumerate(target_names)}).to_numpy()

### 불러온 데이터 타입, 모양, 데이터 확인

print(type(x))

print(type(y))

print(x.shape)

print(y.shape)

print(x)

print(y)

### 학습/테스트 데이터 분할

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

### 트리 분류기 생성 후, fit

breast_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1
)
breast_tree.fit(x_train, y_train)

### 학습/테스트 데이터에 예측 결과 도출 및 평가

y_train_pred = breast_tree.predict(x_train)
y_test_pred = breast_tree.predict(x_test)

acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", acc_train)
print("Test Accuracy:", acc_test)

### Entropy 기준의 트리 분류기 생성 후 fit, 결과 도출, 평가

breast_tree2 = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1
)
breast_tree2.fit(x_train, y_train)

y_train_pred2 = breast_tree2.predict(x_train)
y_test_pred2 = breast_tree2.predict(x_test)

acc_train2 = accuracy_score(y_train, y_train_pred2)
acc_test2 = accuracy_score(y_test, y_test_pred2)

print("Training Accuracy (Entropy):", acc_train2)
print("Test Accuracy (Entropy):", acc_test2)

### Tree 구조의 출력

# Tree 구조 출력을 위한 기본 함수
tree.plot_tree(breast_tree)

plt.figure(dpi=500)
tree.plot_tree(
    breast_tree,
    max_depth=None,
    feature_names=feature_names,
    class_names=target_names,
    filled=True,
    precision=2
)
plt.savefig("./breast_tree.png")
plt.show()

# Titanic data에 대해 실습

data = pd.read_csv("../datasets/titanic.csv")

data

x = data.drop(columns="Survived")
y = data["Survived"]

### One-hot encoding

단, 이전과 같이 4개 범주를 표현하기 위해 3개의 지시 변수를 만들었던 것과 달리 의사결정나무의 분지를 위해서는 4개를 모두 사용

x_encoded = pd.get_dummies(
    x,
    columns=["Sex", "Embarked"],
    drop_first=False
)

### Train/Test를 8:2로 분할 (Testset 20%)

### Train에 대해 의사결정나무 학습

### Train/Test 각각에 대해 예측값 도출

### Train/Test 각각에 대해 성능 계산(accuracy, precision, recall, f1_score, 총 4가지)

x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.2)

dtree = DecisionTreeClassifier(criterion="gini")
dtree.fit(x_train, y_train)



y_train_pred = dtree.predict(x_train)
y_test_pred = dtree.predict(x_test)

acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_test_pred)

print("Training accuracy:", acc_train)
print("Test accuracy:", acc_test)

from sklearn.metrics import precision_score, recall_score, f1_score
prec_train = precision_score(y_train, y_train_pred)
prec_test = precision_score(y_test, y_test_pred)

rec_train = recall_score(y_train, y_train_pred)
rec_test = recall_score(y_test, y_test_pred)

f1_train = f1_score(y_train, y_train_pred)
f1_test = f1_score(y_test, y_test_pred)

print("Training precision:", prec_train)
print("Test precision:", prec_test)
print("Training recall:", rec_train)
print("Test recall:", rec_test)
print("Training F1-score:", f1_train)
print("Test F1-score:", f1_test)

