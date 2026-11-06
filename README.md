# expres-occultation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction
ExPRES-Occultation calculates the opportunities of planetary radio emission sources occultations during flybys of planetary moons. 

The code use ExPRES simulations of planetary radio emissions and WebGeoCalc ephemeris of the spacecraft and the moon.

We have create a set of Tutorial to guide you through the possibilities offered by ExPRES-Occultation:

## Tutorials
- `Tutorial1_plotting_GLL-C09_occultation.ipynb` allows you to simply run the occultation code on the Galileo C09 Flyby of Callisto.
In this tutorial, all required files are already downloaded and stored, and you have nothing else to do than run the tutorial to obtain the occultation plots
- `Tutorial2_plotting_specific_occultation.ipynb` allows you to run the occultation code on any Spacecraft flyby of a moon. The tutorial helps you to go through all the necessary steps to produce the plots.
- `Tutorial3_expres-cosmographia.ipynb` allows you to create the specific files to make Cosmographia movies that will contain the radio emissions sources positions. The tutorial helps you to go through all the necessary steps to produce the files, and explains how to run the scripts in Cosmographia.

## Requirements
### Python version
The code was developper with Python 3.9
### Required packages
The code was developped with the following version of python packages:

numpy==1.26.4
astropy==6.0.1
spacepy==0.7.0
ipykernel