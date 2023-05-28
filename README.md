# RUG-HandRec
Handwriting Recognition

## 1. Create a virtual environment
(make sure you are in the base environemnt)
```
# create a clean virtual environment
conda deactivate
conda env remove -n temp-env-py3.9
conda create -n temp-env-py3.9 python=3.9 -y
conda activate temp-env-py3.9
```

## 2. Install requirements

```
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset


### Hebrew Bible
This dataset was used to train the reconstruction pipeline.

Download the dataset from [http://www.ericlevy.com/Revel/Bible/HebrewOT.pdf](http://www.ericlevy.com/Revel/Bible/HebrewOT.pdf).

to change the conda env use 
```
conda activate <env_name>
```
