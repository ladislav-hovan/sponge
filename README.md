# SPONGE - Simple PriOr Network GEnerator
The SPONGE package generates human prior gene regulatory networks.


## Table of Contents
- [SPONGE - Simple PriOr Network GEnerator](#sponge---simple-prior-network-generator)
  - [Table of Contents](#table-of-contents)
  - [General Information](#general-information)
  - [Features](#features)
  - [Setup](#setup)
  - [Usage](#usage)
  - [Project Status](#project-status)
  - [Room for Improvement](#room-for-improvement)
  - [Acknowledgements](#acknowledgements)
  - [Contact](#contact)
  - [License](#license)


## General Information
This repository contains the SPONGE package, which allows the generation 
of human prior gene regulatory networks based mainly on the data from 
the JASPAR database. It also uses HomoloGene to find the human analogs 
of vertebrate transcription factors, Ensemble to collect all the 
promoter regions in the human genome, UniProt for symbol matching, and
STRING to retrieve protein-protein interactions between transcription
factors.

Prior gene regulatory networks are useful mainly as an input for tools
that incorporate additional sources of information to refine them.
The prior networks generated by SPONGE are designed to be compatible
with PANDA and related [NetZoo](https://github.com/netZoo/netZooPy) 
tools.

The purpose of this project is to give the ability to generate prior 
gene regulatory networks to people who don't have the knowledge or 
inclination to do the genome-wide motif search, but would still like
to change some parameters that were used to generate publicly available
prior gene regulatory networks.

If you just want to use the prior networks generated by the newest 
version of SPONGE with the default settings, they are available on our 
[Google Drive][drive-link].

[drive-link]: https://drive.google.com/drive/folders/16uGpTWFgaFhRDTxSBQ6VtxjqwHY-0fJz?usp=sharing


## Features
The features already available are:
- Generation of prior gene regulatory network
- Generation of prior protein-protein interaction network for
  transcription factors
- Automatic download of required files during setup
- Parallelised motif filtering


## Setup
The requirements are provided in a `requirements.txt` file.


## Usage
A simple workflow would be as follows:

``` python
# Import the class definition
from sponge.sponge import Sponge
# Create the SPONGE object
sponge_obj = Sponge()
# Select the vertebrate transcription factors from JASPAR
sponge_obj.select_tfs()
# Find human homologs for the TFs if possible
sponge_obj.find_human_homologs()
# Filter the matches of the JASPAR bigbed file to the ones in the
# promoters of human transcripts
sponge_obj.filter_matches()
# Retrieve the protein-protein interactions between the transcription
# factors from the STRING database
sponge_obj.retrieve_ppi()
# Write the PPI prior to a file
sponge_obj.write_ppi_prior()
# Aggregate the filtered matches on promoters to genes
sponge_obj.aggregate_matches()
# Write the final motif prior to a file
sponge_obj.write_motif_prior()
```

SPONGE will attempt to download the files it needs into a temporary 
directory (`.sponge_temp` by default). Paths can be provided if these
files were downloaded in advance. The JASPAR bigbed file required for
filtering is huge (> 100 GB), so the download might take some time. Make
sure you're running SPONGE somewhere that has enough space!


## Project Status
The project is: _in progress_.


## Room for Improvement
Room for improvement:
- Add more and better tests

To do:
- Addition of a command line script
- Support for more species


## Acknowledgements
Many thanks to the members of the 
[Kuijjer group](https://www.kuijjerlab.org/) 
at NCMM for their feedback and support.

This README is based on a template made by 
[@flynerdpl](https://www.flynerd.pl/).


## Contact
Created by Ladislav Hovan (ladislav.hovan@ncmm.uio.no).
Feel free to contact me!


## License
This project is open source and available under the 
[GNU General Public License v3](LICENSE).