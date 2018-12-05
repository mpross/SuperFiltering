#!/bin/bash

## Job Name
#SBATCH --job-name=SuperFilt

## Allocation Definition
#SBATCH --partition=stf

## Nodes
#SBATCH --nodes=1

## Walltime
#SBATCH --time=2:00:00

## Memory per node
#SBATCH --mem=100GB

## The working directory for this job
## Make this directory before submitting the job
#SBATCH --workdir=/gscratch/stf/mpross2/SuperFiltering

## Turn on email notification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mpross2@uw.edu

module load anaconda3_4.3.1

python ./superClassify.py




























