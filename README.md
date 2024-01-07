# flightsim-telemetry

## Author: Taylor Morgan

## Installation

Python 3.11 was used for development. The following packages should be installed with pip or similar package management tools:

`pip install plotly pandas numpy scikit-image scikit-learn tensorflow websocket-client threading`

Additionally, to view the training notebook you will need Jupyter, or to import the notebook into a program such as VSCode with notebook support or Google Colab.

## Usage

For visualizing and training the model, open flight_analysis_full.ipynb and run each cell. The required CSV files are included.

For real-time processing of flight data, first open FlightGear and start a flight. Then, run `python listener.py`. 