#!/bin/bash
# Quick experiments to reproduce some results:
python main.py main.mode="full" model="TCN" models.TCN.run_name="1d_regression" preprocessing.multi_dim=False preprocessing.mode="sawtooth" models.TCN.input_size=1 models.TCN.output_size=1
#python main.py main.mode="full" model="TCN" models.TCN.run_name="1d_classification" preprocessing.multi_dim=False preprocessing.mode="binary classification" models.TCN.input_size=1 models.TCN.output_size=2
#python main.py main.mode="full" model="TCN" models.TCN.run_name="65d_classification" preprocessing.multi_dim=True preprocessing.use_magnitude=False preprocessing.mode="binary classification" models.TCN.input_size=65 models.TCN.output_size=2
#python main.py main.mode="full" model="TCN" models.TCN.run_name="65d_regression" preprocessing.multi_dim=True preprocessing.use_magnitude=False preprocessing.mode="sawtooth" models.TCN.input_size=65 models.TCN.output_size=1
#python main.py  main.mode="full"  model="TCN" models.TCN.run_name="130d_regression" preprocessing.multi_dim=True preprocessing.mode="sawtooth" models.TCN.input_size=130 models.TCN.output_size=1
#python main.py  main.mode="full"  model="TCN" models.TCN.run_name="130d_classification" preprocessing.multi_dim=True preprocessing.mode="binary classification" models.TCN.input_size=130 models.TCN.output_size=2


# Main experiments done in the thesis
# Main model: TCN
#python main.py main.mode="leave_one_out" model="TCN" models.TCN.run_name="1d_regression" preprocessing.multi_dim=False preprocessing.mode="sawtooth" models.TCN.input_size=1 models.TCN.output_size=1
#python main.py main.mode="leave_one_out" model="TCN" models.TCN.run_name="1d_classification" preprocessing.multi_dim=False preprocessing.mode="binary classification" models.TCN.input_size=1 models.TCN.output_size=2
#python main.py main.mode="leave_one_out" model="TCN" models.TCN.run_name="65d_classification" preprocessing.multi_dim=True preprocessing.use_magnitude=False preprocessing.mode="binary classification" models.TCN.input_size=65 models.TCN.output_size=2
#python main.py main.mode="leave_one_out" model="TCN" models.TCN.run_name="65d_regression" preprocessing.multi_dim=True preprocessing.use_magnitude=False preprocessing.mode="sawtooth" models.TCN.input_size=65 models.TCN.output_size=1
#python main.py main.mode="leave_one_out" model="TCN" models.TCN.run_name="130d_regression" preprocessing.multi_dim=True preprocessing.mode="sawtooth" models.TCN.input_size=130 models.TCN.output_size=1
#python main.py main.mode="leave_one_out" model="TCN" models.TCN.run_name="130d_classification" preprocessing.multi_dim=True preprocessing.mode="binary classification" models.TCN.input_size=130 models.TCN.output_size=2

# Secondary model: GRU
#python main.py main.mode="leave_one_out" model="GRU" models.GRU.run_name="1d_regression" preprocessing.multi_dim=False preprocessing.mode="sawtooth" models.GRU.input_size=1 models.GRU.output_size=1 preprocessing.slice_duration=30
#python main.py main.mode="leave_one_out" model="GRU" models.GRU.run_name="130d_regression" preprocessing.multi_dim=True preprocessing.mode="sawtooth" models.GRU.input_size=130 models.GRU.output_size=1 preprocessing.slice_duration=30
#python main.py main.mode="leave_one_out" model="GRU" models.GRU.run_name="1d_classification" preprocessing.multi_dim=False preprocessing.mode="binary classification" models.GRU.input_size=1 models.GRU.output_size=2 preprocessing.slice_duration=30
#python main.py main.mode="leave_one_out" model="GRU" models.GRU.run_name="130d_classification" preprocessing.multi_dim=True preprocessing.mode="binary classification" models.GRU.input_size=130 models.GRU.output_size=2 preprocessing.slice_duration=30


# Additional experiments done in the thesis:
# TCN with 5s receptive field
#python main.py model="TCN" models.TCN.run_name="130d_regression_5s_receptive_field" preprocessing.multi_dim=True preprocessing.mode="sawtooth" models.TCN.input_size=130 models.TCN.output_size=1 models.TCN.no_dilation_layers=3
#python main.py model="TCN" models.TCN.run_name="130d_classification_5s_receptive_field" preprocessing.multi_dim=True preprocessing.mode="binary classification" models.TCN.input_size=130 models.TCN.output_size=2 models.TCN.no_dilation_layers=3
#python main.py main.mode="full" models.TCN.run_name="1d_classification_5s_receptive_field" preprocessing.multi_dim=False preprocessing.mode="binary classification" models.TCN.input_size=1 models.TCN.output_size=2 models.TCN.no_dilation_layers=3
#python main.py models.TCN.run_name="1d_regression_5s_receptive_field" preprocessing.multi_dim=False preprocessing.mode="sawtooth" models.TCN.input_size=1 models.TCN.output_size=1 models.TCN.no_dilation_layers=3

#python main.py main.left_out_subject=22 model="TCN" models.TCN.run_name="130d_regression_5s_receptive_field" preprocessing.multi_dim=True preprocessing.mode="sawtooth" models.TCN.input_size=130 models.TCN.output_size=1 models.TCN.no_dilation_layers=3
#python main.py main.left_out_subject=22 model="TCN" models.TCN.run_name="130d_classification_5s_receptive_field" preprocessing.multi_dim=True preprocessing.mode="binary classification" models.TCN.input_size=130 models.TCN.output_size=2 models.TCN.no_dilation_layers=3
#python main.py main.left_out_subject=22 main.mode="full" models.TCN.run_name="1d_classification_5s_receptive_field" preprocessing.multi_dim=False preprocessing.mode="binary classification" models.TCN.input_size=1 models.TCN.output_size=2 models.TCN.no_dilation_layers=3
#python main.py main.left_out_subject=22 models.TCN.run_name="1d_regression_5s_receptive_field" preprocessing.multi_dim=False preprocessing.mode="sawtooth" models.TCN.input_size=1 models.TCN.output_size=1 models.TCN.no_dilation_layers=3

# TCN without preprocessing(raw radar data as input)
#python main.py main.mode="full" model="TCN" models.TCN.run_name="1d_classification_without_preprocessing" preprocessing.multi_dim=False preprocessing.use_magnitude=False preprocessing.apply_peak_envelopes=False preprocessing.apply_butterworth=False  preprocessing.mode="binary classification" models.TCN.input_size=1 models.TCN.output_size=2
#python main.py main.left_out_subject=22 main.mode="full" model="TCN" models.TCN.run_name="1d_classification_without_preprocessing" preprocessing.multi_dim=False preprocessing.use_magnitude=False preprocessing.apply_peak_envelopes=False preprocessing.apply_butterworth=False  preprocessing.mode="binary classification" models.TCN.input_size=1 models.TCN.output_size=2
# TCN without butterworth filter
#python main.py main.mode="full" model="TCN" models.TCN.run_name="1d_classification_without_butterworth" preprocessing.multi_dim=False preprocessing.use_magnitude=False preprocessing.apply_peak_envelopes=True preprocessing.apply_butterworth=False  preprocessing.mode="binary classification" models.TCN.input_size=1 models.TCN.output_size=2
#python main.py main.left_out_subject=22 main.mode="full" model="TCN" models.TCN.run_name="1d_classification_without_butterworth" preprocessing.multi_dim=False preprocessing.use_magnitude=False preprocessing.apply_peak_envelopes=True preprocessing.apply_butterworth=False  preprocessing.mode="binary classification" models.TCN.input_size=1 models.TCN.output_size=2


# Other recurrent models
#python main.py main.mode="full" model="LSTM" models.LSTM.run_name="1d_regression" preprocessing.multi_dim=False preprocessing.mode="sawtooth" models.LSTM.input_size=1 models.LSTM.output_size=1 preprocessing.slice_duration=30
#python main.py main.mode="full" model="RNN" models.RNN.run_name="1d_regression_reshape" preprocessing.multi_dim=False preprocessing.mode="sawtooth" models.RNN.input_size=1 models.RNN.output_size=1 preprocessing.slice_duration=30
#python main.py main.mode="full" model="LSTM" models.LSTM.run_name="1d_regression" preprocessing.multi_dim=False preprocessing.mode="sawtooth" models.LSTM.input_size=1 models.LSTM.output_size=1 preprocessing.slice_duration=30
#python main.py main.mode="full" model="RNN" models.RNN.run_name="1d_regression_reshape" preprocessing.multi_dim=False preprocessing.mode="sawtooth" models.RNN.input_size=1 models.RNN.output_size=1 preprocessing.slice_duration=30

#python main.py main.left_out_subject=22 main.mode="full" model="LSTM" models.LSTM.run_name="1d_regression" preprocessing.multi_dim=False preprocessing.mode="sawtooth" models.LSTM.input_size=1 models.LSTM.output_size=1 preprocessing.slice_duration=30
#python main.py main.left_out_subject=22 main.mode="full" model="RNN" models.RNN.run_name="1d_regression_reshape" preprocessing.multi_dim=False preprocessing.mode="sawtooth" models.RNN.input_size=1 models.RNN.output_size=1 preprocessing.slice_duration=30
#python main.py main.left_out_subject=22 main.mode="full" model="LSTM" models.LSTM.run_name="1d_regression" preprocessing.multi_dim=False preprocessing.mode="sawtooth" models.LSTM.input_size=1 models.LSTM.output_size=1 preprocessing.slice_duration=30
#python main.py main.left_out_subject=22 main.mode="full" model="RNN" models.RNN.run_name="1d_regression_reshape" preprocessing.multi_dim=False preprocessing.mode="sawtooth" models.RNN.input_size=1 models.RNN.output_size=1 preprocessing.slice_duration=30


# Different custom loss functions
#python main.py main.mode="full" model="TCN" models.TCN.run_name="1d_regression_peak_detection_loss" preprocessing.multi_dim=False preprocessing.mode="sawtooth" models.TCN.input_size=1 models.TCN.output_size=1 training.loss_component_weights="{mse: 1.0, peak_detection: 0.1}"
#python main.py main.mode="full" model="TCN" models.TCN.run_name="1d_classification_peak_detection_loss" preprocessing.multi_dim=False preprocessing.mode="binary classification" models.TCN.input_size=1 models.TCN.output_size=2 training.loss_component_weights="{cross_entropy: 1.0, peak_detection_binary: 0.1}"
#python main.py main.mode="full" model="TCN" models.TCN.run_name="1d_regression_derivative_loss" preprocessing.multi_dim=False preprocessing.mode="sawtooth" models.TCN.input_size=1 models.TCN.output_size=1 training.loss_component_weights="{mse: 1.0, derivative: 0.1}"
#python main.py main.mode="full" model="TCN" models.TCN.run_name="1d_regression_second_derivative_loss" preprocessing.multi_dim=False preprocessing.mode="sawtooth" models.TCN.input_size=1 models.TCN.output_size=1 training.loss_component_weights="{mse: 1.0, second_derivative: 0.1}"

#python main.py main.left_out_subject=22 main.mode="full" model="TCN" models.TCN.run_name="1d_regression_peak_detection_loss" preprocessing.multi_dim=False preprocessing.mode="sawtooth" models.TCN.input_size=1 models.TCN.output_size=1 training.loss_component_weights="{mse: 1.0, peak_detection: 0.1}"
#python main.py main.left_out_subject=22 main.mode="full" model="TCN" models.TCN.run_name="1d_classification_peak_detection_loss" preprocessing.multi_dim=False preprocessing.mode="binary classification" models.TCN.input_size=1 models.TCN.output_size=2 training.loss_component_weights="{cross_entropy: 1.0, peak_detection_binary: 0.1}"
#python main.py main.left_out_subject=22 main.mode="full" model="TCN" models.TCN.run_name="1d_regression_derivative_loss" preprocessing.multi_dim=False preprocessing.mode="sawtooth" models.TCN.input_size=1 models.TCN.output_size=1 training.loss_component_weights="{mse: 1.0, derivative: 0.1}"
#python main.py main.left_out_subject=22 main.mode="full" model="TCN" models.TCN.run_name="1d_regression_second_derivative_loss" preprocessing.multi_dim=False preprocessing.mode="sawtooth" models.TCN.input_size=1 models.TCN.output_size=1 training.loss_component_weights="{mse: 1.0, second_derivative: 0.1}"
