#!/bin/bash

apptainer build container.sif container.def
apptainer build auto.sif auto.def

git clone #<anonymized> -> https://anonymous.4open.science/r/sast-BD51
