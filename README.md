# Capital-structure-dml-replication

## Project Overview
Python replication and extension of my MSc dissertation on capital structure determinants using econometric and machine learning methods.

## Research Question
How do firm-level characteristics affect capital structure decisions, and how does a machine learning approach compare with traditional econometric models?

## Methods
- Data cleaning and panel construction
- Fixed effects/ random effects models
- LASSO
- Double Machine Learning (DML)

## Repository Structure
- 'data/': raw and processed data
- 'src/': reusable Python scripts
- 'output/': figures and tables
- 'docs/': methodology and notes

## Current Status
In progress: Currently reconstructing the original dissertation workflow from Stata to Python.

##
During the Python replication process, I verified the source metadata for the macroeconomic controls used in the original Stata workflow and found that some variables had been mislabeled. In particular, the variable labeled as EIR corresponds to annual CPI inflation, and the file labeled as GDP growth corresponds to current account balance as a percentage of GDP. This repository therefore separates the original implementation from a corrected specification based on validated source definitions.

## Author
Lingzhou Wu
