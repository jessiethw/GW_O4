# IceCube Graviational Wave followup of LVK O4

## Overview
This analysis is used to follow up graviational wave events in realtime during the Ligo-VIRGO-Kagra run O4. The analysis searches for common sources of gravitational waves and high-energy neutrinos, by using the gravitational wave localization sent by LVK in realtime as a spatial prior, and doing an all-sky scan for a point source.

## Dependencies
This repository contains scripts used to follow up graviational wave events in realtime, as well as test and validate the analysis codes before the run.
This repository requires a build of IceTray and realtime from the IceCube software (both private). 
The main code framework for this analysis is now part of Fast Response Analysis, https://github.com/icecube/FastResponseAnalysis.
Follow instructions for these pages to set up these codes before running these scripts.

## Note on Confidentiality
Not all relevant code for the analysis is included in the public repository. Much of the code used for this analysis is private, and is therefore not included here. As a result, running some of these notebooks out of the box will not be successful, as paths point to directories on a private server.
