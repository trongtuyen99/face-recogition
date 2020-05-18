from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
# import require lib above

cnn_model = build_model()
#cnn_model.summary()

X, Y = load_data()
le = LabelEncoder()
le.fit(Y)
Y = le.transform(Y)
x_train, x_valid, y_train, y_valid= train_test_split(
    X, Y, test_size=.05, random_state=8888,)

x_train = x_train.reshape(x_train.shape[0], 128, 128, 1)
x_valid = x_valid.reshape(x_valid.shape[0], 128, 128, 1)

history=cnn_model.fit(
    np.array(x_train), np.array(y_train), batch_size=512,
    epochs=150, verbose=2,
    validation_data=(np.array(x_valid),np.array(y_valid)),
)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
