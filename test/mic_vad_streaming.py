import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import queue
import librosa
import onnxruntime
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the ONNX model
onnx_model_path = r"C:\Users\Aamar\Desktop\Project\src\onnx\quanetPrunedCNNLSTMONNX"
ort_session = onnxruntime.InferenceSession(onnx_model_path)
class_labels = ["background", "down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes", "unknown"]

def preprocess_audio(audio_data, num_hops=98, sample_rate=16000, segment_duration=1, frame_duration=0.025, hop_duration=0.010, num_bands=50):
    """
    Preprocesses audio data for inference.
    """
    # Define parameters
    segment_samples = int(segment_duration * sample_rate)
    frame_samples = int(frame_duration * sample_rate)
    hop_samples = int(hop_duration * sample_rate)

    # Pad the audio to a consistent length
    audio_data_padded = np.pad(audio_data, (0, max(0, segment_samples - len(audio_data))), mode='constant')
    audio_data_normalized = librosa.util.normalize(audio_data_padded)

    # Calculate the Bark scale spectrogram
    bark_spectrogram = librosa.feature.melspectrogram(y=audio_data_normalized, sr=sample_rate, n_fft=512, hop_length=hop_samples, n_mels=num_bands)

    # Apply logarithm
    log_bark_spectrogram = np.log10(bark_spectrogram + 1e-6)

    # Truncate or pad the spectrogram to the desired number of hops
    if log_bark_spectrogram.shape[1] > num_hops:
        log_bark_spectrogram = log_bark_spectrogram[:, :num_hops]
    elif log_bark_spectrogram.shape[1] < num_hops:
        pad_width = num_hops - log_bark_spectrogram.shape[1]
        log_bark_spectrogram = np.pad(log_bark_spectrogram, ((0, 0), (0, pad_width)), mode='constant')

    # Transpose the spectrogram to match the expected shape
    log_bark_spectrogram = np.transpose(log_bark_spectrogram)

    # Reshape input data to match the expected shape
    log_bark_spectrogram = np.expand_dims(log_bark_spectrogram, axis=0)  # Add batch dimension
    log_bark_spectrogram = np.expand_dims(log_bark_spectrogram, axis=0)  # Add channel dimension

    return log_bark_spectrogram

def predict(model, audio_data):
    """
    Makes predictions using the provided model.
    """
    # Make predictions
    predictions = model.run(None, {"imageinput": audio_data})
    # Return predictions
    return predictions

class AudioStream:
    def __init__(self):
        self.sample_rate = 16000
        self.blocksize = 20000
        self.q = queue.Queue()
        self.plotdata = np.zeros(self.blocksize)
        self.root = tk.Tk()
        self.root.title("Audio Streaming")

        # Set up styles
        self.style = ttk.Style(self.root)
        self.style.configure('TFrame', background='#FFFFFF')
        self.style.configure('TLabel', background='#FFFFFF', foreground='black', font=('Helvetica', 12))
        self.style.configure('TScale', background='#FFFFFF')
        self.style.configure('TLabelframe', background='#FFFFFF', foreground='black', font=('Helvetica', 14, 'bold'))
        self.style.configure('TLabelframe.Label', background='#FFFFFF', foreground='black')

        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(np.zeros(self.blocksize), color='#0078D7')
        self.ax.axis((0, self.blocksize, -1, 1))
        self.ax.set_yticks(np.linspace(-1, 1, num=5))
        self.ax.yaxis.grid(True, color='gray', linestyle='dashed')
        self.ax.xaxis.grid(False)
        self.fig.patch.set_facecolor('#FFFFFF')
        self.ax.set_facecolor('#FFFFFF')
        self.ax.spines['bottom'].set_color('black')
        self.ax.spines['top'].set_color('black')
        self.ax.spines['left'].set_color('black')
        self.ax.spines['right'].set_color('black')
        self.ax.tick_params(axis='x', colors='black')
        self.ax.tick_params(axis='y', colors='black')

        self.control_frame = ttk.Labelframe(self.main_frame, text="Controls", padding="10")
        self.control_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        self.threshold_label = ttk.Label(self.control_frame, text="Confidence Threshold:")
        self.threshold_label.pack(side=tk.LEFT, padx=(0, 10))
        self.threshold_value_label = ttk.Label(self.control_frame, text="0.50")
        self.threshold_value_label.pack(side=tk.LEFT, padx=(5, 10))
        self.threshold_slider = ttk.Scale(self.control_frame, from_=0, to=1, orient=tk.HORIZONTAL, command=self.update_threshold)
        self.threshold_slider.set(0.5)
        self.threshold_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.amplitude_label = ttk.Label(self.control_frame, text="Amplitude Threshold:")
        self.amplitude_label.pack(side=tk.LEFT, padx=(20, 10))
        self.amplitude_value_label = ttk.Label(self.control_frame, text="0.30")
        self.amplitude_value_label.pack(side=tk.LEFT, padx=(5, 10))
        self.amplitude_slider = ttk.Scale(self.control_frame, from_=0, to=1, orient=tk.HORIZONTAL, command=self.update_amplitude)
        self.amplitude_slider.set(0.3)
        self.amplitude_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.prediction_frame = ttk.Labelframe(self.main_frame, text="Predictions", padding="10")
        self.prediction_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        self.prediction_label = ttk.Label(self.prediction_frame, text="Predicted Class: None")
        self.prediction_label.pack(side=tk.TOP, pady=(0, 5))

        self.prediction_array_label = ttk.Label(self.prediction_frame, text="Prediction Array: []")
        self.prediction_array_label.pack(side=tk.TOP, pady=(0, 10))

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(expand=tk.YES, fill=tk.BOTH)
        self.canvas.draw()

    def audio_callback(self, indata, frames, time, status):
        """
        Audio callback function for real-time streaming.
        """
        if status:
            print(status)
        amplitude = np.max(np.abs(indata))
        if amplitude > self.amplitude_threshold:
            self.q.put(indata.copy())
        self.update_plot()

    def update_plot(self):
        """
        Updates the real-time plot and displays predictions.
        """
        if not self.q.empty():
            data = self.q.get()
            preprocessed_audio_data = preprocess_audio(data[:, 0])
            predictions = predict(ort_session, preprocessed_audio_data)
            print(predictions)
            if predictions:
                prediction = predictions[0][0]
                predicted_class_index = np.argmax(prediction)
                if predicted_class_index < len(class_labels):
                    predicted_class_label = class_labels[predicted_class_index]
                    confidence = prediction[predicted_class_index]
                    prediction_text = f"Predicted Class: {predicted_class_label} (Confidence: {confidence:.2f})"
                    prediction_array_text = f"Prediction Array: {prediction}"

                    self.prediction_label.config(text=prediction_text)
                    self.prediction_array_label.config(text=prediction_array_text)

                    if predicted_class_label != "background" and confidence >= self.confidence_threshold:
                        print("Predicted class:", predicted_class_label)

            self.plotdata = np.roll(self.plotdata, -len(data))
            self.plotdata[-len(data):] = data[:, 0]
            self.line.set_ydata(self.plotdata)
            self.fig.canvas.draw()

    def update_threshold(self, value):
        self.confidence_threshold = float(value)
        self.threshold_value_label.config(text=f"{float(value):.2f}")

    def update_amplitude(self, value):
        self.amplitude_threshold = float(value)
        self.amplitude_value_label.config(text=f"{float(value):.2f}")

    def start_stream(self):
        with sd.InputStream(samplerate=self.sample_rate, blocksize=self.blocksize, channels=1, callback=self.audio_callback):
            self.root.mainloop()

    def stop_stream(self):
        self.running = False

if __name__ == "__main__":
    audio_stream = AudioStream()
    try:
        audio_stream.start_stream()
    except Exception as e:
        print(e)
    finally:
        audio_stream.stop_stream()
