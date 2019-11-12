import snake_game
from random import randint
from generate_training_data import generate_training_data
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


if __name__ == "__main__":
    training_data_x, training_data_y = generate_training_data()
    print('done training data')

    model = Sequential()
    model.add(Dense(units=9,input_dim=7))
    model.add(Dense(units=15,activation='relu'))
    model.add(Dense(units=3, activation='softmax'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    print(len(training_data_x), len(training_data_y))
    model.fit((np.array(training_data_x).reshape(-1,7)),(np.array(training_data_y).reshape(-1,3)), batch_size=512, epochs=6)

    model.save_weights('model.h5')
    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)
