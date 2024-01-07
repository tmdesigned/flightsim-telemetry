import websocket
import threading
import json
import time
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model

class StallPredictor:

    def __init__(self,model_path,scaler_path):
        self.model = load_model(model_path)
        with open(scaler_path, 'rb') as file:
            self.scaler = pickle.load(file)

        self.sequence_length = 1000

        self.pending_ts = 0
        self.last_ts = None
        self.predict_calls = 0

        self.columns = ['pitch', 'roll', 'altitude', 'airspeed']
        self.df = pd.DataFrame(columns = self.columns ).astype(float)
        self.pending_data = {col: None for col in self.columns}


    def update_data(self, data):

        self.pending_ts = data['ts']

         # Update pending_data with new values
        for key, value in data.items():
            if key in self.pending_data:
                self.pending_data[key] = value

        # Check if all values in pending_data are not None
        all_not_none = all(value is not None for value in self.pending_data.values())
        if all_not_none:
            self.add_row()
            self.predict()


    def add_row(self):
        try:
        
            num_to_add = int((self.pending_ts - self.last_ts) / 0.01) if self.last_ts else 1
            self.last_ts = self.pending_ts

            row_to_add = pd.DataFrame([self.pending_data])
            normalized_data = self.scaler.transform(row_to_add)

            # Add new row to end
            new_rows_df = pd.DataFrame(normalized_data.repeat(num_to_add, axis=0), columns=self.columns)
            self.df = pd.concat([self.df, new_rows_df], ignore_index=True)

            # Remove the oldest rows if the DataFrame exceeds a certain size
            max_rows = self.sequence_length  
            if len(self.df) > max_rows:
                self.df = self.df.iloc[len(self.df) - max_rows:]

            # Clear pending data
            self.pending_data = {col: None for col in self.columns}


        except Exception as e:
            # Print the error and exit
            print("Error during transformation:", e)
            exit(1)
    
    def predict(self):
        
        self.predict_calls += 1

        if self.predict_calls % 10 != 0:
            return

        if len(self.df) != self.sequence_length :
            print('waiting...')
            return

        # Reshape for LSTM input
        lstm_input = self.df.values.reshape(1, self.sequence_length, len(self.columns))

        # Make predictions
        predictions = self.model.predict(lstm_input,verbose=0)
        
        if predictions[-1][0] > 0.5:
            print(f"{self.last_ts}: STALLED DESCENT PREDICTED!")
        else:
            print(f"{self.last_ts}: OK")


class WebSocketClient:
    def __init__(self, url, callback):
        self.url = url
        self.ws = websocket.WebSocketApp(url,
                                         on_open=self.on_open,
                                         on_message=callback,
                                         on_error=self.on_error,
                                         on_close=self.on_close)

    def on_error(self, ws, error):
        print("Error:", error)

    def on_close(self, ws, close_status_code, close_msg):
        print("### closed ###")

    def on_open(self, ws):
        def run(*args):
            # Subscribe
            paths = ["/orientation/pitch-deg","/position/altitude-ft","/orientation/roll-deg","/velocities/airspeed-kt"]
            for path in paths:
                ws.send(json.dumps({"command": "addListener", "node": path}))
                ws.send(json.dumps({"command": "get", "node": path}))

        thread = threading.Thread(target=run)
        thread.start()

    def run_forever(self):
        # websocket.enableTrace(True)
        self.ws.run_forever()


def create_callback(predictor):
    def ws_to_predictor_interface(ws, msg):
        name_to_key = {
            "pitch-deg": "pitch",
            "altitude-ft": "altitude",
            "roll-deg": "roll",
            "airspeed-kt": "airspeed"
        }

        data = json.loads(msg)
        name = name_to_key.get(data.get("name"))
        value = data.get("value")
        output = {name: value,"ts":data.get("ts")}
    
        predictor.update_data(output)

    return ws_to_predictor_interface

if __name__ == "__main__":
    predictor = StallPredictor('./stallpredictor.model','./scaler.pkl')
    client = WebSocketClient("ws://localhost:5500/PropertyListener", create_callback(predictor))
    client.run_forever()