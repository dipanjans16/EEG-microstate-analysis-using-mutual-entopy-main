# EEG-microstate-analysis-using-mutual-entopy
This project provides tools to study the relation between different microstates and their non linear interaction with each other using mutual entropy and diffrent dictionary leanning algorithms.

The current research showcases the application of a linear model that incorporates learned EEG maps. We utilized Dictionary Learning algorithms to acquire these EEG maps alongside their corresponding sparse coefficient matrix. Our objective was to identify functionally connected brain states that exhibit substantial differences in connectivity between two distinct populations. To achieve this, we examined the classification accuracy of different online dictionary learning algorithms. We employed both GFP peaks and single source GFP peaks, while also measuring correlations through mutual entropy. To evaluate the effectiveness of our approach, we tested our proposed tool on a dataset comprising individuals categorized as either good or poor task performers.


To understand about EEG microstates and the data that has been used please refer to this research paper-https://www.nature.com/articles/s41598-020-79423-7

All datasets presented in this study are openly available in PhysioNet at https://doi.org/10.13026/C2JQ1P. All epochs that have been preprocessed are available on Figshare at https://doi.org/10.6084/m9.figshare.13135130.

In this project I would like to thank Frederic-vW for provind basic function to extract microstates in the source file https://github.com/Frederic-vW/eeg_microstates . Some of them has been used in this project.
