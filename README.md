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

## Running the Code

### Data Preprocessing
Provide the script with the path to your dataset folder.
Run `python3 data_preprocessing.py`

### Training
Provide the script with the path to your preprocessed dataset folder in the 'config.yaml' file.
There are multiple training configurations available in the 'config.yaml' file.
You also need to provide a wandb API key.

Run `python3 main.py`

### Evaluation
Run `python3 inference_script.py` to create the predictions for the test set.
Then you can run `python3 testing_script.py` to analyze and evaluate the predictions.