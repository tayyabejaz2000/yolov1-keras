from tensorflow.keras.optimizers import Adam

from read_data import BatchedGenerator
from yolo import loss, model

network = model.GetModel((448, 448, 3))

network.compile(optimizer=Adam(learning_rate=1e-4), loss=loss.YOLOLoss())

for batch_x, batch_y in BatchedGenerator(batch_size=256):
    network.fit(batch_x, batch_y, epochs=2,
                batch_size=16, verbose=1)
