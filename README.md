# AT2_experimentation

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project involves building machine learning models to forecast total sales revenue and predict item-level sales at specific stores. The project structure follows the Cookiecutter Data Science template, with models and notebooks categorized into predictive and forecasting subfolders for clarity and organization.

## Project Organization

```
├── README.md          <- Instructions and project overview for developers
├── Makefile           <- Makefile for commands like `make data` or `make train`
│
├── data
│   ├── external       <- Data from third-party sources
│   ├── interim        <- Intermediate transformed data
│   ├── processed      <- Final datasets used for modeling
│   └── raw            <- Original, unaltered data dumps
│
├── docs               <- Documentation for the project (e.g., mkdocs)
│
├── models             <- Trained models and model artifacts
│   ├── predictive     <- Models related to item-level sales prediction
│   └── forecasting    <- Models related to total sales forecasting
│
├── notebooks          <- Jupyter notebooks for experiments and exploration
│   ├── predictive     <- Notebooks for item-level predictions
│   └── forecasting    <- Notebooks for total sales forecasts
│                         Naming: <lastname>_<firstname>-<student_id>-<model_type>_<description>.ipynb
│
├── pyproject.toml     <- Project configuration file (e.g., Poetry for package management)
│
├── references         <- Manuals, data dictionaries, and related resources
│
├── reports            <- Generated analysis reports and visualizations and final report
│   └── figures        <- Generated graphics and figures
│
├── requirements.txt   <- File listing required Python packages for the project
│
├── setup.cfg          <- Configuration file for code style tools like flake8
│
├── github.txt         <- Link to the GitHub repository for experimentation phase
│
└── at2_experimentation   <- Source code for the project
    ├── __init__.py             <- Makes at2_experimentation a Python module
    ├── config.py               <- Configuration variables and settings
    ├── dataset.py              <- Scripts to load, preprocess, and manage data
    ├── features.py             <- Code to create features for the models
    ├── modeling                
    │   ├── __init__.py         
    │   ├── predict.py          <- Scripts for model inference
    │   └── train.py            <- Scripts for training the models
    └── plots.py                <- Functions to create visualizations

```

--------

