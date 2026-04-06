# Capital-Structure-DML-Reconstruction

## Project Overview
This repository reconstructs and extends the empirical workflow of my MSc dissertation on capital structure determinants in Python.

## Research Question
How do firm-level characteristics affect the capital structure decisions of companies, and how do machine learning method such as LASSO and Double Machine Learning differ from traditional econometric approaches?

## Methods
- Panel data construction
- Variable reconstruction and validation
- OLS / fixed effects / random effects models
- LASSO
- Double Machine Learning (DML)

## Repository Structure
- `data/`: raw and processed data
- `src/`: reusable Python scripts
- `output/`: figures and tables
- `docs/`: methodology notes and reconstruction comments

## Important Data Note
During the reconstruction process, I checked the data for the macroeconomic controls used in the original Stata workflow and found that some variables had been mislabeled.

In particular:
- the variable labeled as `EIR` corresponds to annual CPI inflation
- the file labeled as `GDP growth` corresponds to current account balance as a percentage of GDP
- some intermediate merge logic in the original Stata workflow could not be fully verified, which limits exact replication of several original outputs

This repository therefore distinguishes between:
1. the original dissertation implementation, and
2. a corrected specification based on validated source definitions

## Project Goal
The goal of this repository is to reconstruct the dissertation workflow as faithfully as possible, while also providing a corrected and better-documented Python pipeline for subsequent analysis and extension.

## Author
Lingzhou Wu
