# Tariff-Induced Relative Price Shocks and Domestic Substitution

This project implements an instrumental variables (2SLS) analysis of the 2018–2019 Section 301 tariffs to test whether tariffs increased domestic substitution in U.S. manufacturing industries.

## Key Components
- Product-level tariff exposure from Section 301 lists
- Pre-period China import dependence as an instrument
- NAICS 4-digit domestic absorption panel
- First-stage and 2SLS estimation with robustness checks

## Main Finding
Tariffs increased exposure to Chinese import costs but did not generate short-run domestic substitution. Results suggest supply-chain reallocation rather than reshoring.

## Structure
- `code/`: full pipeline scripts
- `archive/`: intermediate/debug scripts
- raw data sourced from U.S. Census and public tariff lists

## Replication
Run scripts sequentially in the `code/` folder to reproduce results.
