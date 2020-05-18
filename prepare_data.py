import os
import numpy as np
import matplotlib.pyplot as plt
def load_data(path_faces = r"/content/face-recogition/face_images"):
  keep_image = 45
  test = np.random.randn(128, 128) # fixed size
  X = np.array(test.reshape(1, *test.shape))
  Y = []
  for folder in os.listdir(path_faces):
    f = os.path.join(path_faces, folder)
    n = len(os.listdir(f)) # min(n) = 49
    Y += [str(folder)] * (45 * 5)
    for file in os.listdir(f)[:45]:
      file_path = os.path.join(f, file)
      img0 = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) # convert to gray scale
      img0 = cv2.resize(img0, (128, 128)) # to make all same size

      img = img0 / 255 #normalize every image
      X = np.concatenate((X, img.reshape(1, *img.shape)))
      # data agumentation

      img_noise = (img0 + 10*np.random.randn(*img0.shape))/255 # add gaussian noise
      X = np.concatenate((X, img_noise.reshape(1, *img0.shape)))

      w = img.shape[1]
      h = img.shape[0]
      M10 = cv2.getRotationMatrix2D((w/2,h/2), 10, 1) # rotate 10
      image1 = cv2.warpAffine(img,M10,(w,h))
      X = np.concatenate((X, image1.reshape(1, *img.shape)))

      M_10 = cv2.getRotationMatrix2D((w/2,h/2), -10, 1) # rotate -10
      image2 = cv2.warpAffine(img,M_10,(w,h))
      X = np.concatenate((X, image2.reshape(1, *img.shape)))
      
      flip = np.fliplr(img) # flip image
      X = np.concatenate((X, flip.reshape(1, *img.shape)))

  X = np.array(X)
  Y = np.array(Y)
  return X[1:], Y

X, Y = load_data()

# X: (2035, (128, 128))
# Y: (2035,)
