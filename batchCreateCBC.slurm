#!/bin/bash

## Job Name
#SBATCH --job-name=SuperFilt

## Allocation Definition
#SBATCH --partition=stf

## Nodes
#SBATCH --nodes=1

## Walltime
#SBATCH --time=4:00:00

## Memory per node
#SBATCH --mem=100GB

## The working directory for this job
## Make this directory before submitting the job
#SBATCH --workdir=/gscratch/stf/mpross2/SuperFiltering

## Turn on email notification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mpross2@uw.edu

module load anaconda3_4.3.1

python ./createData2.py 1 100 1 &
python ./createData2.py 101 200 1 &
python ./createData2.py 201 300 1 &
python ./createData2.py 301 400 1 &
python ./createData2.py 401 500 1 &
python ./createData2.py 501 600 1 &
python ./createData2.py 601 700 1 &
python ./createData2.py 701 800 1 &
python ./createData2.py 801 900 1 &
python ./createData2.py 901 1000 1































