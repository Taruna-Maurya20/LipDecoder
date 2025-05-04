# import os
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import (
#     Conv3D, LSTM, Dense, Dropout, Bidirectional, 
#     MaxPool3D, Activation, Reshape, SpatialDropout3D, 
#     BatchNormalization, TimeDistributed, Flatten
# )
# import os
# from tensorflow.keras.models import Sequential
# # … (your other imports: Conv3D, LSTM, etc.) …

# def load_model() -> Sequential:
#     """Load the pre-trained LipNet model using the .h5 weights exported from Colab."""
#     # Recreate the network architecture exactly as in Colab:
#     model = Sequential()
#     model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same', activation='relu'))
#     model.add(MaxPool3D((1, 2, 2)))
#     model.add(Conv3D(256, 3, padding='same', activation='relu'))
#     model.add(MaxPool3D((1, 2, 2)))
#     model.add(Conv3D(75, 3, padding='same', activation='relu'))
#     model.add(MaxPool3D((1, 2, 2)))
#     model.add(TimeDistributed(Flatten()))
#     model.add(Bidirectional(LSTM(128, return_sequences=True)))
#     model.add(Dropout(0.5))
#     model.add(Bidirectional(LSTM(128, return_sequences=True)))
#     model.add(Dropout(0.5))
#     model.add(Dense(41, activation='softmax'))

#     # **Load the exact .h5 weights you saved in Colab:**
#     weights_path = os.path.join('..', 'models', 'complete_model.weights.h5')
#     model.load_weights(weights_path)
#     print(f"✅ Model weights loaded from {weights_path}")

#     return model


#  upar sahi code hai

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv3D, LSTM, Dense, Dropout, Bidirectional, 
    MaxPool3D, Activation, Reshape, SpatialDropout3D, 
    BatchNormalization, TimeDistributed, Flatten
)

def load_model() -> Sequential:
    """Load the pre-trained LipNet model using the .h5 weights exported from Colab."""
    # Recreate the network architecture exactly as in Colab:
    model = Sequential()
    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same', activation='relu'))
    model.add(MaxPool3D((1, 2, 2)))
    model.add(Conv3D(256, 3, padding='same', activation='relu'))
    model.add(MaxPool3D((1, 2, 2)))
    model.add(Conv3D(75, 3, padding='same', activation='relu'))
    model.add(MaxPool3D((1, 2, 2)))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Dense(41, activation='softmax'))

    # -- FIX THE FILE NAME --
    possible_paths = [
        os.path.join('..', 'models', 'complete_model.h5'),   # <-- correct filename
        os.path.join('models', 'complete_model.h5'),
        os.path.join('.', 'models', 'complete_model.h5'),
    ]

    weights_path = None
    for path in possible_paths:
        if os.path.exists(path):
            weights_path = path
            break

    if weights_path is None:
        raise FileNotFoundError("❌ Could not find 'complete_model.h5' in expected locations.")

    # Load weights
    model.load_weights(weights_path)
    print(f"✅ Model weights loaded from {weights_path}")

    return model
