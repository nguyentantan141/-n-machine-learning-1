# Thư viện hỗ trợ đọc dữ liệu
import os
import numpy as np
import pandas as pd
from skimage import io
from PIL import Image
# Thư viện hỗ trợ tiền xử lí ảnh, dữ liệu
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
# Thư viện hỗ trợ lưu trữ
import joblib
import pickle
# Thư viện  huấn luyện mô hình
from math import sqrt
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
# Trực quan kết quả đánh giá mô hình
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

#Lưu dữ liệu sau khi đã xử lí:
def save_data_image(data,name:str):
  file = open(name, 'wb')
  # dump information to that file
  pickle.dump(data, file)
  # Đóng file
  file.close()

# Lưu mô hình sau khi huấn luyện
def save_model(model,name: str):
  filename= name +'.joblib'
  joblib.dump(model, filename)

#Tiền xử lí ảnh
# Kiểm tra lỗi ảnh
def check_corrupted_image(image_file):
  try:
    with Image.open(image_file) as img:
      img.verify()
      img_new = io.imread(image_file, as_gray=True)
    return False
  except Exception as e:
    print(e)
    return True

# Đọc ảnh
def read_image(path,size,label):
  X=[]
  y=[]
  # Các thư mục có trong file
  files = os.listdir(path)
  # Đọc ảnh xám, resize, làm phẳng
  for image_file in files:
    print(os.path.join(path,image_file))
    if not(check_corrupted_image(os.path.join(path,image_file))):
      img = io.imread(os.path.join(path, image_file), as_gray=True)
      img_vec= resize(img, size).flatten()
      X.append(img_vec)
      y.append(label)
  #y.append(image_file.split('.')[0]) # chỉ sử dụng với tập dữ liệu "train" của kagge
  return X,y


# Đọc dữ liệu lần đầu chuẩn bị để huấn luyện mô hình
def read_new_data():
  size=(32,32)
  path_dog = 'D:/Acer/Dữ liệu/data machine/PetImages/Dog'
  path_cat = 'D:/Acer/Dữ liệu/data machine/PetImages/Cat'
  X,y=read_image(path_dog,size,0)
  X_cat,y_cat=read_image(path_cat,size,1)
  X.extend(X_cat)
  y.extend(y_cat)
  X=np.array(X)
  # Thêm vector cột 1 vào X
  m = X.shape[0]
  X = np.hstack((np.ones((m, 1)), X))
  y=np.array(y)
  y=LabelBinarizer().fit_transform(y)
  #Lưu dữ liệu ảnh đã xử lý
  save_data_image(X,"Dog_cat.txt")
  save_data_image(y,"label_dog_cat.txt")
  return X,y


#Phân chia train-test
def build_img_data(filename_X:str, filename_y:str):
  #X,y=read_new_data()
  # Đọc dữ liệu X
  file_X=open(filename_X,'rb')
  X=pickle.load(file_X)
  file_X.close()
  # Đọc dữ liệu y
  file_y = open(filename_y, 'rb')
  y = pickle.load(file_y).ravel()
  file_y.close()
  X_train, X_test, y_train, y_test=train_test_split(X,y,shuffle=True, test_size=0.2,random_state=100)
  return X_train, X_test, y_train, y_test

# Huấn luyện mô hình Logistic Regression CV
def logistic_regression_cv(X_train, y_train):
  model = LogisticRegressionCV(cv=10, random_state=0,solver='liblinear').fit(X_train, y_train)
  save_model(model,'Logistic_regression_cv')
  return model

# Huấn luyện mô hình K-NN
def k_NN(X_train,y_train):
  # Xác định số lượng mẫu dữ liệu và k_max
  m = y_train.shape[0]
  k_max = int(sqrt(m)/2)
  # Tạo lưới tham số cho GridSearchCV
  k_values = np.arange(start=1, stop = k_max + 1,dtype=int)
  params = {'n_neighbors': k_values}
  # Khởi tạo và huấn luyện mô hình với GridSearchCV
  kNN = KNeighborsClassifier()
  model = GridSearchCV(kNN, params, cv=10)
  model.fit(X_train, y_train)
  save_model(model,'k_NN')
  return model

#Đánh giá mô hình bằng các bộ chỉ số accuracy, precision, recall, f1
def evaluation_model(model_lg,model_knn,X_test,y_test):
  y_hat_lg=model_lg.predict(X_test)
  y_hat_knn=model_knn.predict(X_test)
  #Chỉ số của mô hình logistic
  acc_lg= accuracy_score(y_hat_lg, y_test)
  pre_lg=precision_score(y_hat_lg, y_test)
  rec_lg=recall_score(y_hat_lg, y_test)
  f1_lg=f1_score(y_hat_lg, y_test)
  #Chỉ số của mô hình k-NN
  acc_knn = accuracy_score(y_hat_knn, y_test)
  pre_knn = precision_score(y_hat_knn, y_test)
  rec_knn = recall_score(y_hat_knn, y_test)
  f1_knn = f1_score(y_hat_knn, y_test)
  # Bảng chỉ số của 2 mô hình
  table={'Score':['Accuracy','Precision','Recall','F1'],'Logistic regression':[acc_lg,pre_lg,rec_lg,f1_lg],
         'k-NN':[acc_knn,pre_knn,rec_knn,f1_knn]}
  df=pd.DataFrame(table)
  df.to_csv('Evalution_model.csv', index = False)
  print(df)
  return df

#Vẽ ma trận hỗn hợp
def visual_confusion_matrix(model_lg,model_knn,X_test,y_test):
  y_hat_lg = model_lg.predict(X_test)
  y_hat_knn = model_knn.predict(X_test)
  f, axes = plt.subplots(1, 2)
  #Vẽ ma trận hỗn hợp của mô hình logistic
  cm1 = confusion_matrix(y_test,y_hat_lg)
  lg=ConfusionMatrixDisplay(confusion_matrix=cm1 )
  lg.plot(ax=axes[0])
  lg.ax_.set_title('Logistic regression')
  #Vẽ ma trận hỗn hợp của mô hình k-nn
  cm2= confusion_matrix(y_test,y_hat_knn)
  knn=ConfusionMatrixDisplay(confusion_matrix=cm2 )
  knn.plot(ax=axes[1])
  knn.ax_.set_title('k-NN')
  plt.subplots_adjust(wspace=0.40, hspace=0.1)
  plt.savefig('Confusion matrix.png')
  plt.show()

# Đánh giá mô hình dựa vào Precision & Recall
def visual_precison_recall(model_lg, model_knn,X_test,y_test):
  # Lấy xác suất dự đoán nhãn positive của mô hình
  lg_probs = model_lg.predict_proba(X_test)[:,1]
  knn_probs = model_knn.predict_proba(X_test)[:,1]
  #Lấy nhãn lớp dự đoán và giá trị precision & recall tương ứng
  lg_pre,lg_rec, _ = precision_recall_curve(y_test, lg_probs)
  knn_pre, knn_rec, _ = precision_recall_curve(y_test, knn_probs)
  #Vẽ đường precision - recall
  no_model = len(y_test[y_test == 1]) / len(y_test)
  plt.plot([0, 1], [no_model, no_model], linestyle='--', label='No model')
  plt.plot(lg_rec, lg_pre, marker='.', label='Logistic')
  plt.plot(knn_rec, knn_pre, marker='o', label='k-NN')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.legend()
  plt.title('Precision & recall')
  plt.savefig('precision_recall_image.png')
  plt.show()

# Đánh giá mô hình theo  đường ROC
def visual_roc(model_lg, model_knn,X_test,y_test):
  #Lấy xác suất dự đoán nhãn positive của mô hình
  lg_probs = model_lg.predict_proba(X_test)[:, 1]
  knn_probs = model_knn.predict_proba(X_test)[:, 1]
  #Tính ROC score
  lg_auc = roc_auc_score(y_test, lg_probs)
  knn_auc = roc_auc_score(y_test, knn_probs)
  print('Mô hình Logistic Regression - ROC AUC: ', lg_auc)
  print('Mô hình k-NN - ROC AUC: ', knn_auc)
  #Vẽ đường ROC
  lg_fpr, lg_tpr, _ = roc_curve(y_test, lg_probs)
  knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
  plt.plot(lg_fpr, lg_tpr, marker='.', label='Logistic')
  plt.plot(knn_fpr, knn_tpr, marker='3', label='k-NN')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.legend()
  plt.title('ROC')
  plt.savefig('roc.png')
  plt.show()

def main():
  filename_X='Dog_cat.txt'
  filename_y='label_dog_cat.txt'
  X_train, X_test, y_train, y_test=build_img_data(filename_X,filename_y)
  #huấn luyện mô hình
  #logistic_regression_cv(X_train,y_train)
  #k_NN(X_train,y_train)
  #Tải mô hình đã lưu
  lg_model=joblib.load('Logistic_regression_cv.joblib')
  knn_model=joblib.load('k_NN.joblib')
  #đánh giá và trực quan hóa kết quả
  visual_precison_recall(lg_model,knn_model,X_test,y_test)
  visual_roc(lg_model,knn_model,X_test,y_test)
  evaluation_model(lg_model,knn_model,X_test,y_test)
  visual_confusion_matrix(lg_model,knn_model,X_test,y_test)

if __name__=='__main__':
  main()
