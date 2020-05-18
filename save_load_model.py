from keras.models import model_from_json
def save_model(cnn_model)

    model_json = cnn_model.to_json()
    with open("/content/drive/My Drive/model_92.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    cnn_model.save_weights("/content/drive/My Drive/model_92.h5")
    print("Saved model to disk")
     
    # later...
def load_model(path_h5="/content/drive/My Drive/model_92.h5", path_json='/content/drive/My Drive/model_92.json'):
    # load json and create model
    json_file = open(, 'r')
    loaded_model_json = json_file.read(path_json)
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path_h5)
    print("Loaded model from disk")
