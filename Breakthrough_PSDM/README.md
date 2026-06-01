# Breakthrough PSDM Smoke Test Guide

This project prepares experimental PFAS breakthrough data for fixed-bed treatment modeling. The workflow is built around concepts used in adsorption and ion-exchange design, especially the fixed-bed breakthrough ideas discussed in **MWH’s Water Treatment: Principles and Design, Third Edition**.

In Chapter 15, the book describes adsorption systems using terms such as **breakthrough profile**, **treatment objective**, **empty-bed contact time (EBCT)**, **specific throughput**, and **carbon usage rate**. In Chapter 16, the book discusses ion-exchange fixed beds, including **EBCT**, **film diffusion**, **intraparticle diffusion**, **resin capacity**, and breakthrough behavior during loading. These ideas guide how the breakthrough class is organized in this project.

The smoke test confirms that the data reader, breakthrough class, Excel files, plotting, and prepared output files are connected correctly.

---

# Terminal Commands

## Move into the project folder

After downloading or cloning the repository, open Terminal and move into the project folder:

(cd path/to/Breakthrough_PSDM)

## Optional: activate the virtual environment
source .venv/bin/activate

## Install required packages
pip install pandas openpyxl matplotlib

## Run the smoke test through the terminal
python smoke_test.py