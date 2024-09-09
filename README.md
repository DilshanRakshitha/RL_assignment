# CartPole Reinforcement Learning Example
This project implements a basic Reinforcement Learning agent to solve the CartPole environment using the OpenAI Gymnasium library. The agent uses Q-learning to balance a pole on a cart.

## Requirements
Make sure you have Python 3.11 installed.

## Running the CartPole Example

Clone this repository
Activate your environment
If you are using a virtual environment or conda, activate it:
```
conda activate your_env_name    # for conda users
```
#### OR
```
source venv/bin/activate         # for virtualenv users
```
then install the required packages using pip install -r requirements.txt

## Run the training script:

To start training the agent and running the CartPole example, run:
```
python cartpole.py
```
The script will train the Q-learning agent in the CartPole environment.

## Viewing Results:

After training, the agent will try to solve the CartPole task by keeping the pole balanced for as long as possible. The training results, such as total rewards and the number of episodes, will be printed in the terminal.
run the code
```
python validation_cartpole.py
```