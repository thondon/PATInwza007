import os, time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
FILE_CODE = "4"
DATA_DIR = f"./data/{FILE_CODE}/"
LATENT_DIM = 16
REGULARIZATION_L2 = 0.2
ALPHA = 1
BETA = 1009
DIFF_RATE = 10037
SAMPLE_RATE = 2200  # Approx sample rate
GRAPH_SCALE = (0, 0.25)
FRAME_SIZE = 10  # Segment size in frames
THREEHOLD = 1
# ----------------------------------------

def load_data(directory, prefix="t"):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".npy") and f.startswith(prefix)]
    data = [np.load(f) for f in files]
    reshaped_data = [d.reshape(1, -1) for d in data]
    return np.array(reshaped_data)

def normalize_data(data):
    epsilon = 1e-8
    return (data - np.min(data)) / (np.max(data) - np.min(data) + epsilon)

class LSTM_Autoencoder(tf.keras.Model):
    def __init__(self, input_shape, latent_dim):
        super(LSTM_Autoencoder, self).__init__()
        self.encoder = keras.Sequential([
            layers.InputLayer(shape=input_shape),
            layers.LSTM(128, activation="sigmoid", return_sequences=True, kernel_regularizer=keras.regularizers.l2(REGULARIZATION_L2)),
            layers.LSTM(64, activation="sigmoid", return_sequences=False, kernel_regularizer=keras.regularizers.l2(REGULARIZATION_L2)),
            layers.Dense(latent_dim, activation="linear"),
        ])
        self.decoder = keras.Sequential([
            layers.RepeatVector(input_shape[0]),
            layers.LSTM(64, activation="sigmoid", return_sequences=True, kernel_regularizer=keras.regularizers.l2(REGULARIZATION_L2)),
            layers.LSTM(128, activation="sigmoid", return_sequences=True, kernel_regularizer=keras.regularizers.l2(REGULARIZATION_L2)),
            layers.TimeDistributed(layers.Dense(input_shape[1]))
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

# --- Load and train autoencoder ---
train_data = normalize_data(load_data(DATA_DIR, prefix="t")) * BETA
# test_data = normalize_data(load_data(DATA_DIR, prefix="e")) * BETA  # Load test data
test_data = normalize_data(load_data(f"./data/4/", prefix="e")) * BETA  # Load test data
input_shape = train_data.shape[1:]

def rmse_metric(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

model = LSTM_Autoencoder(input_shape, LATENT_DIM)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.Huber(delta=1.0),
    metrics=[rmse_metric]
)
history = model.fit(train_data, train_data, epochs=50, batch_size=64, validation_split=0.2, callbacks=[
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
])

# --- Plot training loss ---
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.ylim(GRAPH_SCALE)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

def segment_by_frames(waveform, frame_size):
    segments = []
    # Loop over the waveform and create segments of fixed frame size
    for start in range(0, len(waveform) - frame_size + 1, frame_size):
        segments.append(waveform[start:start + frame_size])
    return segments

def analyze_segments(train_input_segment, output_segment):
    # Calculate energy for the input segment (training data) using sum of squared values
    energy_input = np.sum(np.square(train_input_segment))
    
    # Calculate energy for the output segment (model's reconstruction of test data)
    energy_output = np.sum(np.square(output_segment))
    
    # Difference between input and output (error)
    diff = (energy_input - energy_output) * DIFF_RATE
    note = "✅ ปกติ" if abs(diff) < THREEHOLD else "เบาไป" if diff < 0 else "แรงไป"
    
    return energy_input, energy_output, diff, note

# --- Evaluate test files ---
for idx, test_sample in enumerate(test_data):
    os.system("cls")
    print(f"\n📄 ตรวจไฟล์: Test {idx + 1}")

    test_input = test_sample.reshape(1, 1, -1)
    predicted_output = model.predict(test_input, verbose=0)[0][0] * ALPHA
    output_segments = segment_by_frames(predicted_output, FRAME_SIZE)

    best_match = {
        'train_index': -1,
        'diff_sum': float('inf'),
        'mse': float('inf'),
        'mae': float('inf'),
        'train_prediction': None,
        'train_segments': None,
        'min_diff': float('inf')
    }

    for j, train_sample in enumerate(train_data):
        train_input = train_sample.reshape(1, 1, -1)
        predicted_train = model.predict(train_input, verbose=0)[0][0] * ALPHA
        train_segments = segment_by_frames(predicted_train, FRAME_SIZE)
        tmp_main_diff = float('inf')

        # Compute per-segment absolute difference
        total_diff = 0
        for train_segment, output_segment in zip(train_segments, output_segments):
            _, _, diff, _ = analyze_segments(train_segment, output_segment)
            total_diff += abs(diff)
            tmp_main_diff = min(tmp_main_diff, diff)

        # Global metrics (optional)
        mse = np.mean((predicted_output - predicted_train) ** 2)
        mae = np.mean(np.abs(predicted_output - predicted_train))

        if total_diff < best_match['diff_sum']:
            best_match.update({
                'train_index': j,
                'diff_sum': total_diff,
                'mse': mse,
                'mae': mae,
                'train_prediction': predicted_train,
                'train_segments': train_segments,
                'min_diff': tmp_main_diff
            })

    # Use best match to analyze per-segment comparison
    print(f"\n🧠 Best Match: Train Sample {best_match['train_index']} (Total Diff={best_match['diff_sum']:.4f})")

    for i, (train_segment, output_segment) in enumerate(zip(best_match['train_segments'], output_segments)):
        input_energy, output_energy, diff, note = analyze_segments(train_segment, output_segment) #, best_match['min_diff'])
        print(f"พยางค์ {i+1}: จริง={input_energy:.4f}, สร้าง={output_energy:.4f}, ต่าง={diff:.4f} → {note}")
        time.sleep(0.1)

    print(f"\nGlobal MSE for Test {idx + 1}: {best_match['mse']:.4f}, Global MAE: {best_match['mae']:.4f}")

    plt.figure(figsize=(12, 3))
    plt.plot(predicted_output, label="Predicted Output (Test Data)")
    plt.plot(best_match['train_prediction'], label=f"Best Train Match (Train {best_match['train_index']})", linestyle='--')
    plt.title(f"Test {idx + 1}: Predicted Test vs Best Predicted Train")
    plt.legend()
    # plt.show()

    time.sleep(5)
