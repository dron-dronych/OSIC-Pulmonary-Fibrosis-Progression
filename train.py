import utils.imutils as iut
import utils.utils as utils
import argparse
import pandas as pd


def setup_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', help='path to base directory with CT scans',
                    required=True
                    )
    ap.add_argument('-m', '--model', help='path to save output model')

    return vars(ap.parse_args())

def read_fvc_data(path):
    return pd.read_csv(path)


if __name__ == '__main__':
    args = setup_cli()
    BASE_DIR = args['dataset']
    model_save_path = args['model']

    fvc_train = read_fvc_data(BASE_DIR + 'train.csv')

    # get rid of bad images of this patient
    fvc_train = fvc_train[fvc_train['Patient'] != 'ID00052637202186188008618']

    TRAIN_IMG_DIR = BASE_DIR + 'train/'
    train = iut.prepare_dataset(fvc_train, TRAIN_IMG_DIR, train=True)

    model = utils.build_model(
        (500, 500)
    )

    history = model.fit(train, epochs=2, steps_per_epoch=len(fvc_train))

    if model_save_path:
        model.save(model_save_path, save_format='h5')
