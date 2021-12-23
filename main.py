import os

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from read_data import BatchedGenerator
from yolo import loss, model

model_checkpoint = ModelCheckpoint(
    'model.h5',
    save_weights_only=True
)


network = model.GetModel((448, 448, 3))

network.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=loss.YOLOLoss()
)

if 'model.h5' in os.listdir(os.getcwd()):
    network.load_weights('model.h5')

for batch_x, batch_y in BatchedGenerator(batch_size=512):
    network.fit(
        batch_x,
        batch_y,
        epochs=2,
        batch_size=32,
        callbacks=[model_checkpoint]
    )
