# A 2D/3D Polarization Mapping Program for Piezoelectric Force Microscopy (PFM)
This is a series of programs to obtain 2d/3d map of polarization informaion measured from PFM. The raw data should be in Amplitude & Phase pairs on each point of the sample.
## The environment requirements are listed below:
+ Python 3.7

## A brief introduction to the source codes are listed below
+ 'vector_map.py' is the main program to generate vector map.
+ 'phase_analyze.py' is the program to pretreat the phase data.
+ 'error_analyze.py' is the program to map the fitting error on each point.
+ 'gauss.py' is an independent program to fit the phase or amplitude data stastically by gauss function.
+ 'export_field.py' is a program to export 2D or 3D mapping data as EXCEL format.

## Other files are listed beloew
'data/' directory contains raw data from PFM as example. 'result demo/' shows what the final result would be like. 'ref/' contains the reference.

Hope this project can help you, and your contributions are always cherished.
