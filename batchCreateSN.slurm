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

python ./createData.py 1 10 1 &
python ./createData.py 11 21 1 &
python ./createData.py 21 30 1 &
python ./createData.py 31 40 1 &
python ./createData.py 41 50 1 &
python ./createData.py 51 60 1 &

python ./createData.py 61 70 1 &
python ./createData.py 71 80 1 &
python ./createData.py 81 90 1 &
python ./createData.py 91 100 1 &
python ./createData.py 101 110 1 &

python ./createData.py 111 120 1 &
python ./createData.py 121 130 1 &
python ./createData.py 131 138 1 
##python ./createData.py 141 150 1 &
##python ./createData.py 151 160 1 &

##python ./createData.py 161 170 1 &
##python ./createData.py 171 180 1 &
##python ./createData.py 191 200 1 &
##python ./createData.py 201 210 1 &
##python ./createData.py 211 220 1 &

##python ./createData.py 221 230 1 &
##python ./createData.py 231 240 1 &
##python ./createData.py 241 250 1 &
##python ./createData.py 251 260 1 &
##python ./createData.py 261 270 1






























