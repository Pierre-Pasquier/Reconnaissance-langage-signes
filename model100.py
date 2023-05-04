import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

train_df = pd.read_csv("datas/archive/sign_mnist_train.csv")
test_df = pd.read_csv("datas/archive/sign_mnist_test.csv")

test = pd.read_csv("datas/archive/sign_mnist_test.csv")
y = test['label']


train_df.head()


plt.figure(figsize = (10,10)) # Label Count
sns.set_style("darkgrid")
sns.countplot(train_df['label'])




















