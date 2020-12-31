# lab_intro_notebooks
Jupyter notebooks for learning basics of python, MNE-python, STRF-fitting and more. Tutorials are more specific to neuroscience applications in the lab.

## Getting Started
Welcome to the lab! If you're reading this, hopefully you are getting started learning how to use python and jupyter notebooks to do some analysis of neuroscience data! In our lab, this includes scalp electroencephalography (EEG) as well as intracranial electrophysiology (ECoG or stereo EEG). The tutorials here are organized into Jupyter notebooks that will go through a series of concepts that will give you the foundation for what you need in the lab. Of course, there are many more excellent tutorials that are broader and go into more detail on some topics that we won't address. I'll leave links for those at the bottom. For now, start with reading this README, and then choose which notebook you would like to start with!

### Why python?
Python is a great choice for scientific computing, has a wide user base in academia and industry, and is free! Skills you learn from python can be transferred to other programming languages relatively easily, but python is still one of the most popular programming languages across many fields.

To run these notebooks on your own computer, I recommend installing [Anaconda python version 3, 64 bit](https://www.anaconda.com/products/individual#Downloads). If you have trouble running these notebooks on your own computer, you can try running them on [Google colab](https://colab.research.google.com/), which won't require you to install anything on your computer, however, there may be code packages you'll have to download from within the colab session. Ask Liberty if you need help.

### What is a Jupyter notebook?
From the [Project Jupyter](https://jupyter.org) website: 
> The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, machine learning, and much more.

## List of notebooks
### 01_intro_jupyter_notebook.ipynb
Introduction to jupyter notebooks, when to use them, when not to use them.

### 02_intro_python
Introduction to basic python that is used frequently in lab scripts and analysis. This includes an introduction to data types, data structures, loading csv or text files, array manipulation, list comprehensions, and more.

### 03_MNE_preprocessing_EEG
Includes loading data, referencing, filtering, and performing artifact rejection using ICA.

### 04_MNE_preprocessing_ECoG
Includes loading data, referencing, manual artifact rejection, and filtering/extraction of high gamma band activity

### 05_MNE_plotting
After you've preprocessed your data, this gives some ways of plotting evoked data, topographic maps, and more.

### 06_preprocessing_audio
This provides an intro to some tools that we use to extract different acoustic features, such as the envelope, pitch, formants, or spectrogram of a sound.

### 07_forced_aligner
This provides an intro on how to use the FAVE align forced aligner to automatically annotate the phoneme and word boundaries in a sound file. We use this to obtain a Praat TextGrid file, which can be used in later epoching and data analysis.

### 08_textgrid_manipulation
Intro on how to create phoneme matrices, phonological feature matrices, and more from textgrids.

### 09_mTRF_analysis
How to use ridge regression to fit a multivariate temporal receptive field (mTRF). We use our own python code to do this, with ridge regression code optimized for speed on the types of neural data matrices we work with. Also includes variance partitioning, noise ceiling correction.

### 10_neural_networks
Intro to using convolutional neural networks in encoding model frameworks.

### 11_tacc
General advice on using TACC resources (Texas Advanced Computing Center)

## Other resources
* [Codecademy python 2](https://www.codecademy.com/learn/learn-python) -- (python 3 is also available, but is not free. There are some minor differences in syntax but the tutorial for python 2 is definitely still useful if you're just starting out).
* From UC Berkeley's [Computational Models of Cognition course](https://github.com/compmodels)
  * [Linear algebra in python](https://github.com/compmodels/problem-sets/blob/master/linear_algebra_review/Linear%20Algebra%20in%20Python.ipynb)
  * [Principal component analysis](https://github.com/compmodels/problem-sets/blob/master/pca_demo/PCA%20Demo.ipynb)
  * [Review of probability](https://github.com/compmodels/problem-sets/tree/master/probability_review)
