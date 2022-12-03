import os
import numpy as np
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
      img = io.imread(image_file, as_gray=True)
    return False
  except Exception as e:
    print(e)
    return True
# Đọc ảnh
def read_image(path,size):
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
      y.append(image_file.split('.')[0])

  return X,y
def main():
  size=(32,32)
  path_dog = 'D:/Acer/Dữ liệu/data machine/PetImages/Dog'
  path_cat = 'D:/Acer/Dữ liệu/data machine/PetImages/Cat'
  X,y=read_image(path_dog,size)
  X_cat,y_cat=read_image(path_cat,size)
  X.extend(X_cat)
  y.extend(y_cat)
  X=np.array(X)
  y=LabelBinarizer().fit_transform(y)
  X_train, X_test, y_train, y_test=train_test_split(X,y,shuffle=True, random_state=100)
  return X_train, X_test, y_train, y_test
if __name__== '__main__':
  main()
# import pickle, lưu dữ liệu lại