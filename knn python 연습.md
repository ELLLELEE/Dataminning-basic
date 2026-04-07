# knn python 연습
### - knn 데이터 
- 우린 knn으로 분류 모델을 학습시키려고 함
- 그러기 위해선 우린 데이터의 모든 값들을 숫자로 바꿔줄 필요가있음
1)  df_copy_with_dummy = pd.get_dummies(
    
    data = df_copy,
    
    columns = ['Sex', 'Embarked'],
    
    drop_first = True
  
)
- 그 다음 열의 상태가 bool인걸 전부 따로 모으기
2) df_subset_bool = df_copy_with_dummy.select_dtypes('bool')

- 따로 빼놓은 bool 값들을 숫자로 바꿔줘야됨
3) df_subset_float = df_subset_bool.astype(float)

- 따로 빼놓은 값 숫자값들을 다시 원래 있던 데이터 프레임에 넣어야됨
4) df_copy_with_dummy[df_subset_bool.columns] = df_subset_float

### - knn 모델 학습
- 우리가 만든 데이터 프레임에서 종속 변수, 독립변수 구분
1) x = df_copy_with_dummies.drop["Survived", axis=1)
2) y = df_copy_with_dummies["Survived"]

- knn 모델 가져오기
3) from sklearn,neighbors import KNeighborsClassifier
4) knn = KNeighborsClassifier(n_neighbor=10)

- 모델 학습 + y_pred 구하기
5) knn.fit(x,y)
6) y_pred = knn.predict(x)

- 여기서 y_pred는 넘파이 형태로 출력
- 그래서 계산하기해서 x,y둘다 넘파이형식으로 바꿔줘야함

7) x = x.to_numpy() ; y = y.to_numpy()

### - knn 평가
- 우리는 실제 값인 y와 예측값인 y_pred의 값의 차이를 알고 싶
1) from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
2) print("Accuracy:", accuracy_score(y, y_preds))
   
  print("Precision:", precision_score(y, y_preds))
  
  print("Recall:", recall_score(y, y_preds))
  
  print("F1 score:", f1_score(y, y_preds))

#### * confusion+matrix and confusionMatrixDisplay

### - 데이터 전처리
1) from sklearn.preprocessing import StandardScaler
2) x_copy = np.copy(x)
3) scaler = StandardScaler()
4) scaler.fit(x_copy)
5) x_std = scaler.transform(x_copy)
6) knn_std.fit(x_std, y)
7) y_std_pre = knn_std.predict(x_std)
8) 
