# Stochastic-Hill-Sliding-Climbing
This is a script for normal stochastic hill climbing, it optimizes a given function by iteratively exploring the solution space. The optimization process involves generating random perturbations to the current solution and accepting or rejecting these perturbations based on their impact on the objective function.

# How to use it

## 1. Create a virtual environment:
```python
python -m venv .venv
```

the activate you virtual environment:
Run:

```python
.venv\Scripts\Activate
```
But if you dont see (venv) in bracket in your terminal, run:

```python
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process; .venv\Scripts\Activate
```

The virtual environment will allow you to run the project is a environment with all the dependecies need.

## 2. Clone github repo

Next clone the github repo, this will download all the needed files into your workspace:

```python
git clone https://github.com/KaraboLetsholo/Stochastic-Hill-Sliding-Climbing.git
```

## 3. Install the Requirements

Install all the need dependecies and requirements like JAX and wandb.

```python
pip install -r requirements.txt
```

## 4. Run the script

```
python Stochastic_Hill_Climbing
```
