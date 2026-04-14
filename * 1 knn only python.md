Numpy와 Pandas 설치가 필요한 경우, 바로 아래 칸 코드 실행

!pip install numpy pandas

import numpy as np
import pandas as pd

Machine learning 알고리즘 실행을 도와주는 scikit-learn 라이브러리 설치를 위해 아래 칸 코드 실행

!pip install scikit-learn

### Understanding kNN classifiers

data = pd.read_csv("/content/drive/MyDrive/dataset/titanic.csv")
print(data.head())

from google.colab import drive
drive.mount('/content/drive')

만약 위 코드 실행 시, 파일 혹은 경로를 발견할 수 없다는 에러(No such file or directory)가 발생하면,

(1) 현재 실행 경로 혹은 그 아래 폴더에 데이터를 옮겨넣거나,

(2) 괄호 속에 데이터 파일의 절대 경로를 찾아 넣어야 함.

# 데이터 이해 및 전처리
print(data.info())

data

#### 데이터 이해를 위한 정보

Pclass: 1 = 1등석, 2 = 2등석, 3 = 3등석

Sex: male = 남성, female = 여성

Age: 나이

SibSp: 동승한 자매, 배우자의 수

Parch: 동승한 부모, 자식의 수

Fare: 승객 요금

Embarked: 탑승지

Survived: 0 = 사망, 1 = 생존

#### 전처리

scikit-learn의 kNN 분류기는 사용할 하나의 거리 척도를 정해주어야 한다(예: 유클리드 거리, 맨해튼 거리 등)

연속형, 범주형 변수가 섞여있는 데이터(Mixed-type data)는 처리하지 못함.

# copy한 dataframe을 사용
# 만약 원래 데이터셋을 그대로 사용하거나, df_copy = data을 사용하면?
# -> 원래 데이터셋이 변하게 되면서 잘못 처리된 경우 되돌리기 어려움.
# knn을 쓸땐 무조건 전처리가 필요
df_copy = data.copy()

df_copy

# category variables -> numerical variables
#{사과, 배, 딸기} 범주형 변수 : one-hot-encoding, dummy variables(더미 변수)
# 사과 -> [1,0,0] , 배 -> [0,1,0] , 딸기 -> [0,0,1]
# 사과 -> [0,0], 배-> [1,0], 딸기 -> [0,1]
df_copy_with_dummies = pd.get_dummies(
    data=df_copy,
    columns=["Sex", "Embarked"],
    drop_first=True
)

print(df_copy_with_dummies.head())

df_subset_bool = df_copy_with_dummies.select_dtypes("bool")
print(df_subset_bool.head())

df_subset_float = df_subset_bool.astype(float)
print(df_subset_float.head())

print(df_subset_bool.columns)

df_copy_with_dummies[df_subset_bool.columns] = df_subset_float

# same operation
# df_copy_with_dummies.loc[:, df_subset_bool.columns] = df_subset_float

print(df_copy_with_dummies.head())

# kNN 모델 학습 및 평가

from sklearn.neighbors import KNeighborsClassifier

data_proc = df_copy_with_dummies
# 여기서는 그냥 할당해도 됨. 데이터를 더 변형시킬 것이 아니니까.

# 독립 변수, 종속 변수를 구분하는 방법
print(data_proc.drop("Survived", axis=1))

print(data_proc["Survived"])

x = data_proc.drop("Survived", axis=1)
y = data_proc["Survived"]

knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(x,y)

y_preds = knn.predict(x)

print(y_preds)

print(type(y_preds))

x = x.to_numpy()
y = y.to_numpy()

#### 분류기의 평가

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy:", accuracy_score(y, y_preds))
print("Precision:", precision_score(y, y_preds))
print("Recall:", recall_score(y, y_preds))
print("F1 score:", f1_score(y, y_preds))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

conf_mat = confusion_matrix(y, y_preds)

conf_mat

ConfusionMatrixDisplay.from_predictions(y, y_preds)

# Another approach
ConfusionMatrixDisplay.from_estimator(knn, x, y)

ROC-AUC?

일반적인 kNN과 같이 단순히 예측 클래스만을 반환하면, ROC 커브를 그릴 수 없음(결정 경계에 따라 예측이 변하지 않음).

kNN에서 ROC 커브를 그리고 싶다면, 확률 예측값을 얻어내야 함. How?

y_probs = knn.predict_proba(x)
print(y_probs)

print(y_probs.shape)

from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(y, y_preds)

# area under the curve
from sklearn.metrics import auc
roc_auc = auc(fpr, tpr)
print(roc_auc)

import matplotlib.pyplot as plt

# draw plot.
fig = plt.figure(figsize=(15, 9))

plt.plot(fpr, tpr, "b", label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "r--")

plt.xlim([0, 1])
plt.ylim([0, 1])

plt.title("Receiver Operating Characteristic Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend(loc="lower right")

plt.show()

# Standardize

from sklearn.preprocessing import StandardScaler

x_copy = np.copy(x)
print(x_copy)

scaler = StandardScaler()
scaler.fit(x_copy)

x_copy

x_std = scaler.transform(x_copy)
print(x_std)

knn_std = KNeighborsClassifier(n_neighbors=10)
knn_std.fit(x_std, y)

y_std_pred = knn_std.predict(x_std)

print("Accuracy:", accuracy_score(y, y_std_pred))
