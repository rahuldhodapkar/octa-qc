# File Descriptions
Overview of files within the repository and associated data descriptions

## src

Contains source code for classifiers and statistics to evaluate them. Python scripts are
designed to be used with a [virtualenv](https://docs.python.org/3/library/venv.html)
and are tested with python version `3.8.5`. Package requirements are included in a standard
`requirements.txt` file.

Note for running code on Yale HPC (Farnam) - tested with:

    module load HTSeq/0.13.5-foss-2020b-Python-3.8.6

and added to environment in `.bash_profile`:

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ysm-gpfs/apps/software/CUDAcore/11.1.1/lib64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ysm-gpfs/apps/software/cuDNN/8.0.5.39-CUDA-11.1.1/lib64

Sample usage:

    virtualenv venv
    . ./venv/bin/activate
    python --version # should be 3.8.6
    pip install -r requirements.txt


R scripts have been tested on R version 4.0.3 (2020-10-10) -- "Bunny-Wunnies Freak Out".
Dependencies will need to be installed independently prior to running code.

## BuildMLClassifier.py

## InterRaterBenchmarks.R
Generate figures to compare ratings from Rater 1 (Jay Wang) with Rater 2 (Rahul Dhodapkar)

## Data

### metadata
Data containing sample characteristics for each OCT-A scan.

Data is structured in CSV format as follows:

| Column Name        | Data                 |
| ------------------ | -------------------- |
| ID | De-identified OCT-A image ID |
| Age | Patient age at image acquisition |
|Sex | Patient sex (0 = Male, 1 = Female) |
| HBA1C | Last HbA1c value before scan |
| HBA1CToScanTime | Months between HBA1C and OCT-A scan |
| Ethnicity | Ethnicity (0 = Caucasian, 1 = Black, 2 = Hispanic, 3 = Asian, 4 = Other) |
| Eye | Eye laterality (0 = OD, 1 = OS) |
| SignalStrength | Signal Strength value from Zeiss AngioPlex |
| DMStage | Stage of Diabetic Retinopathy (N/A = could not be determined, NONE = no DR, MILD = mild NPDR, MOD = moderate NPDR, S = severe NPDR, E = early PDR, HR = high risk PDR) |
| VesselDensity | [Vessel Density](https://pubmed.ncbi.nlm.nih.gov/26803800/) in [0, 1] |
| SkeletonizedVesselDensity | [Skeletonized Vessel Density](https://pubmed.ncbi.nlm.nih.gov/30339262/) in [0, 1]  |

### quality_ratings
Data sets containing manual quality control ratings for testing and training
of machine learning algorithms.

Rater #1 := Jay Wang, MD
Rater #2 := Rahul Dhodapkar, BS Computer Science

Data is structured in CSV format as follows:

| Column Name        | Data                 |
| ------------------ | -------------------- |
| ID | De-identified OCT-A image ID |
| Decentered | 0 = centered, 1 = >20px deviation of fovea from center of image |
| BlinkLine | 0 = none, 1 = non-significant (<20 pixels), 2 = significant (>=20 pixels) |
| Quilting | 0 = none, 1 = slight, 2 = moderate, 3 = significant |
| VesselDoubling | # quadrants |
| Displacement | # quadrants |
| StretchArtifacts | # quadrants |
| MAS | Motion artifact score - see manuscript |
| SAS | Segmentation artifact score - see manuscript |
| MediaOpacity | 0 = none, 1 = <1/16, 2 = <1/4, 3 = >1/4 |
| SmallCapillariesNotVisible | 0 = small capillaries visible, 1 = small capillaries not visible |
| Gradable | 0 = no, 1 = borderline, 2 = yes - see manuscript |
| StrictGradable | 0 = no, 1 = yes |
