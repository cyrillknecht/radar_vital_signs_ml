# Generative Machine Learning for Radar-Based Vital Sign Monitoring

This repository contains the code for the Semester Thesis "Generative Machine Learning for Radar-Based Vital Sign Monitoring" by Cyrill Knecht.
The Thesis was conducted at the Project Based Learning Lab at ETH Zurich.

## Project Description
Vital sign monitoring is fundamental in current medicine and contact-less monitoring
is being developed as it is preferred over traditional methods in scenarios
where patients cannot attach electrodes to 
their body or contact sensors are intrusive and can spread infections.
New low-power radar sensors ear being researched as a means to operate touchless, unintrusive,
and privacy-respecting continuous monitoring applications,
with the potential to revolutionize preventive healthcare in hospitals and at home.
Existing work has proven the feasibility of heart-rate extraction with traditional signal processing methods.
While these methods are effective under certain scenarios, they have robustness
issues.
The goal of this project is to investigate machine-learning algorithms to increase the quality of the signal,
by removing artifacts and reconstructing the pulses starting from the radar data.

# Running the Code

## Prerequisites
- Python 3.7
- pip

## Installation
1. Clone the repository
2. Install the requirements with `pip install -r requirements.txt`

## Execution
There are three possible ways to run the code:
1. The runner bash script `run_experiments.sh`
2. Running the `main.py` file
3. Running the pipeline components individually [`preprocessing.py`, `training.py`, `inference.py`, `testing.py`]

Both methods will run the code with the default parameters.
The parameters can be changed in the `config.yaml` file in the `configs` directory.
Before running the code, the user has to set up the wandB API key in the `config.yaml` file in the `configs` directory.
This can be done by setting the `wandb.api_key` parameter to the API key of the user.

### 1. The runner bash script
The runner bash script is a simple bash script that runs the main.py file with the given arguments.
The script is located in the root directory of the repository and can be ran with the following command:
```shell
  bash run_experiments.sh
```
The runner bash script has lots of experiments that can be run by uncommenting the corresponding lines in the script.
By default the script runs the following command:
```bash
  python main.py main.mode="full" main.model="TCN" models.TCN.run_name="1d_regression" preprocessing.multi_dim=False preprocessing.mode="sawtooth" models.TCN.input_size=1 models.TCN.output_size=1
```
A breakdown of the command is the following:
- `main.mode="full"`: The mode of the code. The possible values are `full`, `leave_one_out`,`train`, `test`.
- `main.model="TCN"`: The model to be used. The possible values are `TCN`, `GRU`, `LSTM`, `RNN`.
- `models.TCN.run_name="1d_regression"`: The name of the run. The name of the run is used to save the results of the run to wandB.
- `preprocessing.multi_dim=False`: Whether to use multi-dimensional input or not.
- `preprocessing.mode="sawtooth"`: The mode of the preprocessing. The possible values are `sawtooth` or `binary classification`.
- `models.TCN.input_size=1`: The input dimension of the model (1,65,130).
- `models.TCN.output_size=1`: The output size of the model(1,2).

For a full breakdown of the parameters, please look into the Configurations section.


### 2. Running the main.py file
The main.py file is the entry point of the code. It can be run with the following command:
```shell
  python main.py
```
All the configuration parameters are also overwritable with command line arguments.
An example of running the main.py file with command line arguments is the following:
```shell
python main.py main.mode="full" main. model="TCN" models.TCN.run_name="1d_regression" preprocessing.multi_dim=False preprocessing.mode="sawtooth" models.TCN.input_size=1 models.TCN.output_size=1
```

### 3. Running the pipeline components individually
The pipeline components can be run individually as scripts.
Run the following command to run the preprocessing pipeline:
```shell
  python preprocessing.py
```
This will preprocess the raw data in the `data` directory and save the data that
was preprocessed according to your configurations
in .h5 files the `preprocessed_data` directory.
These directories can also be adjusted in the configuration file.
The test set and validation set can be specified in the configuration file.
(`main.left_out_subjects`(int) and `main.val_subjects`(list of int) to be chosen from range [0,24])


Run the following command to run the training:
```shell
  python training.py
```
This will train your specified model with the available preprocessed data according to your configurations.
The trained model will be saved in the directory
specified in the configuration file(usually the `outputs` directory).
The training script will also save the results of the training to wandB.
If the data dimensions are not compatible with the model, the script will raise an error.

Run the following command to generate inference with a trained model:
```shell
  python inference.py
```

This will generate the output of the model on the test data and save the results in the `outputs` directory.
Make sure to specify the path to the model to be used in the configuration file under `inference.model_path`
(for example `outputs/mymodel.ckpt`).
The results of the inference will be saved in the same directory
as your preprocessed data(specified in the configuration file under `dirs.data_dir`)
as results.csv(previous results will be deleted before saving the new results).

Run the following command to run the evaluation:
```shell
  python testing.py
```
This will run the evaluation of on a results.csv file
and log the results to wandB.



## Configurations
The configurations of the code are stored in the `config.yaml` file in the `configs` directory.
The hydra package is used to manage the configurations.

The `config.yaml` file contains the following sections:
- `wandb`: The wandB configuration.
```yaml
wandb:
  project_name: "CHOOSE_YOUR_WANDB_PROJECT_NAME"
  api_key: "YOUR_WANDB_API_KEY"
```
WandB is used as an integral part of this project to log the results of the training and testing to the wandB dashboard.
To disable this or choose another type of logging,
the source code has to be changed.

- `dirs`: The directories used in the code to store and access data.
```yaml
dirs:
  data_dir: "YOUR_RAW_DATA_DIR_PATH"
  unprocessed_data_dir: "YOUR_UNPROCESSED_DATA_DIR_PATH"
  save_dir: "outputs"
```
The `data_dir` is the directory where the raw radar dataset is stored.
The `unprocessed_data_dir` is the directory where the preprocessed data and the results of the inference should be stored.
The `save_dir` is the directory where the trained models shall be saved.

- `model`: The model to be used.
```yaml
model: TCN
```
Possible values are 'TCN' for Temporal Convolutional Network,
'GRU' for Gated Recurrent Unit,
'LSTM' for Long Short-Term Memory and
'RNN' for Recurrent Neural Networks.
The configuration of the model can be further specified in the `models` section.

- `models`: The configurations of the models.
```yaml
  TCN:
    run_name: "test_1d_regression"
    input_size: 1 
    output_size: 2 
    hidden_size: 200
    num_layers: 11 
    kernel_size: 3
    no_dilation_layers: 0

  GRU:
    ...
```
The `run_name` is the name of the run. The name of the run is used to save the results of the run to wandB.
The `input_size` is the input dimension of the model (1,65,130).
This is the number of features of the input data.
It has to be adjusted according to the input data.
The `output_size` is the output size of the model(1,2).
This is the number of features of the output data.
It has to be adjusted according the target signal.
The `hidden_size` is the number of features in the hidden state of the model.
The `num_layers` is the number of layers in the model. Each layer has size `hidden_size`.
The `kernel_size` is the size of the kernel in the convolutional layers. Only used in the TCN model.
The `no_dilation_layers` is the number of layers in the model that do not have dilation. This is used only in the TCN model and only for a specific experiment.
In this experiment sing num_layers=11 and no_dilation_layers=3 results in a receptive field of roughly 5 seconds for the TCN model
differing from the default 30 second receptive field width.

- `preprocessing`: The configurations of the preprocessing function/script.
```yaml
preprocessing:
  multi_dim: False
  mode: "sawtooth" 
  slice_duration: 30 
  apply_butterworth: True 
  apply_peak_envelopes: True 
  use_magnitude: True


```
The `multi_dim` parameter specifies whether we want to use 130 input features or only 1(True or False).
This can also be adjusted further by setting the `use_magnitude` parameter to False which results in a 65 feature input for multi_dim=True.
The inputs either consist of the processed phase signal of the most likely radar bin(1),
the processed phase signal of all radar bins(65),
or the magnitude signal and processed phase signal of all radar bins(65).
The extent of the preprocessing can be adjusted by setting the `apply_butterworth` and `apply_peak_envelopes` parameters to True or False.
The `mode` parameter specifies the form of the target signal. The possible values are `sawtooth` or `binary classification`.
The sawtooth signal generates a signal of shape (1,2954) for 30 seconds of data. It is a continuous signal with a sawtooth shape. The sawtooth peaks
are the R-peaks of the original ECG target signal.
The binary classification signal generates a signal of shape (2,2954) for 30 seconds of data.
It is a one-hot encoded signal with the first channel being the R-peaks and the second channel being the non-R-peaks.
The `slice_duration` parameter specifies the duration of the slices of the input data. The input data is sliced into slices of this duration.
Generally, the slice duration should be set to 30 seconds.

- `training`: The configurations of the training function/script.
```yaml
training:
  max_epochs: 1
  batch_size: 8
  learning_rate: 1e-3
  dev_mode: False 
  early_stopping_patience: 5
  loss_component_weights:
    # regression losses
    mse: 1
    derivative: 0
    second_derivative: 0
    peak_detection: 1
    # classification losses
    cross_entropy: 1
    peak_detection_binary: 1
```
The `max_epochs` parameter specifies the maximum number of epochs to train the model.
The training will stop after this number of epochs if Early Stopping does not stop it earlier.
The `batch_size` parameter specifies the batch size of the training data.
The `learning_rate` parameter specifies the learning rate of the optimizer.
The `dev_mode` parameter specifies whether to run the code in development mode or not. 
In development mode the training will be minimal for debugging purposes. This only works when using the 'train' mode.
The `early_stopping_patience` parameter specifies the patience of the early stopping. The training will stop if the validation loss does not improve for this number of epochs.
The `loss_component_weights` parameters specify the weights of the different loss components.
For regression tasks, the loss components are:
- `mse`: The mean squared error loss.
- `derivative`: The L1 loss of the first derivative of the signal.
- `second_derivative`: The L1 loss of the second derivative of the signal.
- `peak_detection`: An approximation of the difference between
the number of peaks of the signal and the number of peaks of the target signal.
Scaled to approximately the same order of magnitude as the other loss components.

For classification tasks, the loss components are:
- `cross_entropy`: The cross-entropy loss.
- `peak_detection`: An approximation of the difference between the number of peaks of the signal and the number of peaks of the target signal.
Scaled to approximately the same order of magnitude as the other loss components.


The weights of the loss components can be adjusted to change the importance of the different loss components.
The loss components are then added together to form the total loss of the model.

- `inference`: The configurations of the inference function/script.
```yaml
inference:
  model_path: ""
```
The `model_path` parameter specifies the path to the model checkpoint to be used for the inference.
Make sure that a model with this name does not already exist in the `save_dir` directory before running a full training and testing run.
- `testing`: The configurations of the testing function/script.
```yaml
testing:
  plot: True 
  prominence: 0.2
  wandb_log: True
```
The `plot` parameter specifies whether to plot the results of the testing or not. It works in conjunction with the `wandb_log` parameter.
If `wandb_log` is set to True and `plot` is set to True, the results of the testing will be logged to wandB and the plots will be saved to WandB.
If 'wandb_log' is set to False and `plot` is set to True, only the first 10 results will be analyzed and plotted(to avoid problems with too many matplotlib figures).
These plots will also be saved in a 'plots' directory.
The `prominence` parameter specifies the prominence parameter of the scipy find_peaks function. It is used to find the peaks in the signal.
Changing it will result in different results of the testing.
```yaml
main:
    mode: "full"
    preprocess_data: False
    left_out_subject: 1 
    val_subjects: [0] 
```
The `mode` parameter specifies the mode of the code. The possible values are `full`, `leave_one_out`,`train`, `test`.
full: Runs the full pipeline. Preprocesses the data, trains the model, generates inference and runs the testing.
leave_one_out: Runs the full pipeline 24 times(once for each patient in the data set) with leave one out cross-validation.
train: Only preprocesses the data and trains the model.
test: Only generates inference and runs the testing. Needs a model to be specified in the `inference.model_path` parameter.
The `preprocess_data` parameter specifies whether to preprocess the data or not.
If set to False, the data will not be preprocessed again. Useful for multiple runs in full mode with the same preprocessed data.
In leave_one out mode preprocess_data is set to True at runtime to avoid mistakes.
The `left_out_subject` parameter specifies the subject to be left out in the 'full or `train` modes.
The `val_subjects` parameter specifies the subjects to be used as the validation set in all modes.
This was set to [0] for all the experiments in the Thesis.


## Repository Structure

The repository is structured as follows:
- `configs`: The directory containing the configuration files. Contains only the `config.yaml` file.
A different configuration file could also be used. One would have to change the configuration file in the `main.py` file.
```python
@hydra.main(version_base="1.2", config_path="configs", config_name="NAME_OF_NEW_CONFIG_FILE")
def main(cfg: DictConfig):
    ...
```
One would also need to change this in all the pipeline components when trying to run them individually.

- `helpers`: The directory containing the helper functions.
This directory contains preprocessing and data loading functions provided by the Thesis supervisors.
- `models`: The directory containing the models.
This directory contains the models used in the Thesis implemented as PyTorch torch.nn.Modules. 
It also contains a generic Pytorch Lightning module that can be used as a wrapper to train any model.
and a get_model function that returns the model specified in the configuration file.

- `pipeline`: The directory containing the data pipeline components.'
'preprocessing.py': Implements the loading the data from the raw data directory, preprocessing the data and saving the preprocessed data.
The preprocessing 
'dataloader.py': The data loading pipeline component. It loads the preprocessed data from h5 files and creates a PyTorch DataLoader.
'training.py': The training pipeline component. It trains the model with the preprocessed data using the PyTorch Lightning Trainer. The best model is saved in the `save_dir` directory.
'inference.py': The inference pipeline component. It generates the output of the model on the test data and saves the results.
'testing.py': The testing pipeline component. It runs the evaluation of the results and logs the evaluation results to wandB.


