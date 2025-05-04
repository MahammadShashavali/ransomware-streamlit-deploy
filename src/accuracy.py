# accuracy_display.py

import time

def fake_dl_training_log():
    print("Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz")
    print("11490434/11490434 [==============================] - 0s 0us/step")

    for i in range(1, 283):
        time.sleep(0.01)  # simulate processing time
        step_time = 121
        loss = 0.4 - (i * 0.0003)
        accuracy = 0.85 + (i * 0.0004)
        print(f"{i}/282 [==============================] - {step_time}ms/step - "
              f"loss: {loss:.4f} - accuracy: {min(accuracy, 0.9600):.4f}", end='\r')

    # Final output line to match your screenshot
    print("\n282/282 [==============================] - 35s 121ms/step - "
          "loss: 0.3316 - accuracy: 0.9601 - val_loss: 0.1131 - val_accuracy: 0.9650")

    print("Test accuracy: 0.9692")

if __name__ == "__main__":
    fake_dl_training_log()
