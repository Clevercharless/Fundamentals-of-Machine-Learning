import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from sklearn.preprocessing import LabelEncoder

# Sample dataset
texts = [
    "I am very happy", "feeling sad", "so angry", "what a joy", "I am depressed",
    "totally furious", "I'm feeling great", "heartbroken", "this is amazing", "very upset"
]

emojis = ["😊", "😢", "😡", "😊", "😢", "😡", "😊", "😢", "😊", "😡"]  # Labels

# Encode emojis to numeric labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(emojis)  # ["😊", "😢", "😡"] → [2, 0, 1]

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, padding='post')

# Define the model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=16, input_length=padded.shape[1]),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')  # 3 emoji classes
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(padded, np.array(labels), epochs=50, verbose=0)

# Test
test_text = input("Enter something: ")
test_seq = tokenizer.texts_to_sequences(test_text)
test_pad = pad_sequences(test_seq, maxlen=padded.shape[1], padding='post')
pred = model.predict(test_pad)
pred_emoji = label_encoder.inverse_transform([np.argmax(pred)])
print(f"Input: {test_text[0]} → Emoji: {pred_emoji[0]}")
