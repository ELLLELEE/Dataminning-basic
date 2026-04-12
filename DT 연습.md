# DT 연습 파이썬 코드
## - 데이터 셋 가져오기
1) from sklearn.datasets import load_breast_cancer
2) breast = load_breast_cancer()
3) x = breast.data ; y = breast.target

## - 데이터 셋 스플릿
1) import sklearn.model_selection import train_test_split
2) x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, stratify = y)

## - tree 모델 가져오기
1) from sklearn import tree
2) from sklearn.tree import DecisionTreeClassifier

## - 모델 학습시키기 지니 모델을 기반으로
1) breast = DecisionTreeClassifier(
         criterion = "gini",
         max_depth = None,
         min_smaples_split = 2,
         min_samples_leaf = 1
   )
2) breast.fit(x_train, y_train)

## - 모델 정확도 측정
1) from sklearn.metrics import accuracy_score, confusion_matri
2) y_train_pre = breast_tree.predict(x_train)
3) y_test_pre = breast_tree.predict(x_test)
4) acc_train = accuracy_score(y_train_pre, y_train)
5) acc_test = accuracy_score(y_test_pre, y_test)

## - 모델 엔트로피 기반으로 학습 시키기
- breast2 = DecisionTreeClassifier(
          criterion = "entropy",
          max_depth = None,
          min_samples_split = 2,
          min_samples_leaf = 1
  )

## - 트리 시각화
- 기본 문법
  - tree.plot_tree(breast_tree)
1) plt.figure(dpi=500) -> 도화지 만들기
2) tree.plot_tree(
   breast_tree,
   max_depth=None,
   feature_names = feature_names,
   class_names = target_names,
   filled = True,
   precision = 2
   )
3) plt.show()

## - 타이다닉 데이터 셋으로 연습
1) data -> to think
2) x = > data.drop(columns = "Survived")
3) y = > data["Survived")
#### -> 데이터 인코딩 

->

x_encoded = pd.get_dummies(

    x,
    columns = ["Sex", "Embarked"],
    
    drop_first=False
)

4) dtree = DecisionTreeClassifier(criterion="gini")
