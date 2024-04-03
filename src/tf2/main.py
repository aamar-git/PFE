import tf2

model = tf2.load_model()
model.save("qPrunedCNNLSTM.keras")
model.save("qPrunedCNNLSTM.h5")
