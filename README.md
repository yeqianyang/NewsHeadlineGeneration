# NLP final project

This folder contains the code for my NLP final project.

There are three files:

- `prepare_data_multi_process.py`: This file processes the data from the Sougou Dataset and convert it into format that will be read in by the trainer.
- `title_generation.py`: This file trains a Seq2Seq model on the preprocessed data.
- `test.py`: This file generate title using the trained model.


## Prerequirments

The code is based on the Tensorflow deep learning library. To run it, you would need to install Tensorflow first.

## Training

To train the model, type the following command:

    python title_generation.py -lr 0.001 --data sougou_business_sport.pkl
