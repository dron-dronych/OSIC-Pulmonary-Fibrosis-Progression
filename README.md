# OSIC Pulmonary Fibrosis Progression
## Predicting lung function decline

We are tasked with a problem to predict severity of a lung function decline based on a computed tomography (CT) scan.

### Dataset
Open Source Imaging Consortium (OSIC) has provided chest scans and accompanying patient information. A patient has an image acquired at time Week = 0 and has numerous follow up visits over the course of approximately 1-2 years, at which time their FVC (forced vital capacity (FVC), i.e. the volume of air exhaled) is measured.

### Pipeline
This ML endeavor is particularly targeted at using Tensorflow's tf.data API due to abundance of image data supplemented with tabular patient data with subsequent feature generation techniques. This all makes extensive use of in-memory computations which buries the idea of inference on slower machines.

### Usage
```
python3 train.py --dataset [PATH_TO_BASE_DIR]
