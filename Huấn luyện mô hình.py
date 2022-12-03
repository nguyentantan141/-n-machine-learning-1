import os
import numpy as np
import pickle
from skimage import io
from skimage.transform import resize
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import pickle
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

#Lưu dữ liệu sau khi đã xử lí:
def save_data_image(data,name:str):
  file = open(name, 'wb')
  # dump information to that file
  pickle.dump(data, file)
  # Đóng file
  file.close()

# Đọc ảnh
def read_image(path,size,label):
  X = []
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
  #y.append(image_file.split('.')[0]) # chỉ sử dụng với tập dữ liệu "train" của kagge
  y=[label for _ in range(len(X))]
  return X,y

# Đọc dữ liệu lần đầu chuẩn bị để huấn luyện mô hình
def read_new_data():
  size=(32,32)
  path_dog = 'D:/Acer/Dữ liệu/data machine/PetImages/Dog'
  path_cat = 'D:/Acer/Dữ liệu/data machine/PetImages/Cat'
  X,y=read_image(path_dog,size,'D')
  X_cat,y_cat=read_image(path_cat,size,'C')
  X.extend(X_cat)
  y.extend(y_cat)
  X=np.array(X)
  y=LabelBinarizer().fit_transform(y)
  #Lưu dữ liệu ảnh đã xử lý
  save_data_image(X,"Dog_cat.txt")
  save_data_image(y,"label_dog_cat.txt")
  return X,y

def main():
  X,y=read_new_data()
  # Đọc dữ liệu X
  file_X=open('Dog_cat.txt','rb')
  X=pickle.load(file_X)
  file_X.close()
  # Đọc dữ liệu y
  file_y = open('label_dog_cat.txt', 'rb')
  y = pickle.load(file_y)
  file_y.close()
  print(X)
  print(y)
  X_train, X_test, y_train, y_test=train_test_split(X,y,shuffle=True, random_state=100)
  return X_train, X_test, y_train, y_test
if __name__== '__main__':
  print(main())
