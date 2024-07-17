# PointNet: A method for nuclei segmentation with Points Annotation
## Description
This page includes the code corresponding to our method.


## Dependencies
This code runs in an environment with PyTorch version 1.8.0 and CUDA version 11.2.



## Usage
### Prepare data
* First, the images (*images*), instance annotations (*labels_instance* stored in *png* format), 
 and the JSON file containing the dataset split (*train_val_test.json*) need to be stored in the following structure:
   
      data
      |-- dataset
      | |-- images
      | |-- labels_instance
      | |-- train_val_test.json
* Next, to generate data for training, run the code:
```bash
python prepare_data.py
```

### Train
You only need to run `main.py` to execute the code. The parameter settings can be adjusted in `options.py` or `main.py`.
```train
python main.py 
```

### Test
You can directly run `test.py` for testing with default parameters. Additionally, some commands can be directly input in the command line, such as:
```
python test.py --model-path ./experiments/monuseg/checkpoints/checkpoint_60.pth.tar
```
