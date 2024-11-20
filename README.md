# Galaxy Morphology and Evolution Pertaining to Asymmetry
## Abstract
Since its launch, the James Webb Space Telescope (JWST) has repeatedly brought into question our understanding of 
the evolution of the universe. The goal of our project is to study the evolution of galaxies by measuring structural 
parameters such as the asymmetry and sizes of galaxies around z=1 and compare these same parameters in nearby galaxies. 
Using the JWST images of the SMACS 0723 field observed with the NIRCam camera and F115W filter we have sampled 
40 galaxies between z=0.6 and z=1.4. To do the necessary calculations, we have written in-house Python code to 
measure the asymmetry, a measurement that is sensitive to the choice of the center of the galaxy. We have already 
tested two methods to measure the asymmetry, and we are exploring additional methods of measuring their centers with 
SExtractor. Our measured asymmetry values will be compared with the Petrosian Radii from Woods et al., 2024. 
These two structural parameters will give us a better comparison with nearby galaxy samples as the JWST observations 
map the same restframe wavelength as those from the SDSS. In addition, we will directly compare our measurements with 
those obtained using the Hubble Space Telescope in the visible. We expect to measure smaller differences in the 
structural parameters when we compare them in the same rest frame (i.e. when comparing the NIR parameters of z=1 
galaxies with those measured in the visible of local galaxies). We will also determine the difference in the asymmetry 
when measured with NIRCam versus with HST of similar populations of z=1 galaxies.

--------------------------
## Developers
* Developer: [Mekhi Woods]() (mekhidw@hawaii.edu)
* Developer: [Augustus Coffey]() (atcoffey@hawaii.edu)
* Advisor: [Marianne Takamiya]() (takamiya@hawaii.edu)

--------------------------
## Operation Guide
1. Install necessary dependencies:
   * Install necessary packages by running `pip install -r requirements.txt`
2. Place FITS files of field galaxies in `fits\`
3. Place target list in main directory.
4. Replace file pathways with location of FITS and target file `main.py` main function.
5. Run `main.py`
--------------------------
## Petrosian Radius

--------------------------
## Asymmetry

--------------------------
## Publications
Please cite the following if this repo is utilized for a scientific project:
* 
--------------------------
## Dependencies
* python ( >= 3.9 )
* matplotlib==3.9.2 
* numpy==2.1.1 

