import pandas as pd
import numpy as np
import os

vocabulary = {'_': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8,
              'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16,
              'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24,
              'y': 25, 'z': 26}

def process_xyt(x, y, t, max_abs_value):
    # shift all the times to zero
    t = t - t[0]

    # shift x and y values to (0, 0) and scale
    x_scaled = np.array([(a - x[0]) / max_abs_value for a in x])
    y_scaled = np.array([(a - y[0]) / max_abs_value for a in y])

    return pd.Series([x_scaled, y_scaled, t])

def calculate_v_a_r(x, y, t):
    # calculates the speed in units/ms, acceleration in units/ms/ms and approach from previous point between -pi and pi
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)

    distance = np.sqrt(dx**2 + dy**2)
    speed = distance / dt  # in units/ms
    padded_speed = np.concatenate(([speed[0]], speed))

    dv = np.diff(padded_speed)
    acceleration = dv / dt  # in units/ms
    padded_acceleration = np.concatenate(([acceleration[0]], acceleration))

    angles = np.arctan2(dy, dx)  # in radians
    padded_angles = np.concatenate(([angles[0]], angles))

    return pd.Series([padded_speed, padded_acceleration, padded_angles])

def collate_input(x_coord, y_coord, time, speed, accel, angle):
    return [[x, y, t, v, a, r] for x, y, t, v, a, r in zip(x_coord, y_coord, time, speed, accel, angle)]

def process_target(word):
    word = word.lower()
    indices = [vocabulary[char] for char in word]
    # one hot encode the characters
    encoded = [np.eye(27, dtype=int)[idx] for idx in indices]
    return encoded

def process_data(file_path):
    df = pd.read_json(file_path, lines=True, encoding="utf-16")
    # remove usernames as this is not needed
    df = df.drop(columns=["username"])
    # remove any datapoints where there are less than 5 datapoints
    df = df[df['swipe'].apply(lambda x: isinstance(x, list) and len(x) > 5)].reset_index(drop=True)

    # Create new columns for x, y, and t as numpy arrays
    df['x'] = df['swipe'].apply(lambda swipe: np.array([point[0] for point in swipe]))
    df['y'] = df['swipe'].apply(lambda swipe: np.array([point[1] for point in swipe]))
    df['t'] = df['swipe'].apply(lambda swipe: np.array([point[2] for point in swipe]))

    # find the maximum values of x and y
    flattened_x = np.concatenate(df['x'].values)
    flattened_y = np.concatenate(df['y'].values)
    max_abs_value = max(np.max(np.abs(flattened_x)), np.max(np.abs(flattened_y)))

    # process x y and t values
    df[["x", "y", "t"]] = df.apply(lambda row: process_xyt(row["x"], row["y"], row["t"], max_abs_value), axis=1)
    # calculate speed, acceleration and angle of approach
    df[["velocity", "acceleration", "angles"]] = df.apply(lambda row: calculate_v_a_r(row['x'], row['y'], row['t']), axis=1)

    # put the inputs back into a single datapoint
    df["input"] = df.apply(lambda row: collate_input(row['x'], row['y'], row['t'], row["velocity"], row["acceleration"], row["angles"]), axis=1)
    
    # tokenise the target word for CTC loss
    df["target"] = df["word"].apply(process_target)

    # only return the word, the input and target
    return df[["word", "input", "target"]]

if __name__ == "__main__":
    file_path = os.path.join(os.getcwd(), "dataset", "swipes.ndjson")
    df = process_data(file_path)
    print(df)

    # for combining multiple data files later
    # result_df = pd.concat([df1, df2], ignore_index=True)

    save_path = os.path.join(os.getcwd(), "processed_data", "data_1.parquet")
    df.to_parquet(save_path)
