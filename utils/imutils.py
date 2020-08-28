import random
import tensorflow_io as tfio
import tensorflow as tf


IMG_RESIZE = [500, 500]
BATCH_SIZE = 16
BUFFER_SIZE = 16


def load_dicom(img_path):
    """
    load image in DICOM format
    """
    dataset = pydicom.dcmread(img_path)

    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)

        print('Image size: {}x{}'.format(
            rows, cols
        ))

    plt.imshow(dataset.pixel_array)


def load_random_dicoms(dicom_dir, n_imag=5, seed=22):
    """"""
    random.seed(seed)  # for reproducability
    counter = 0

    imgs = glob.glob(dicom_dir + '/*/*.dcm')
    plt.figure(figsize=(14, 14))

    while counter < n_imag:
        im = random.choice(imgs)

        plt.subplot(2, 3, counter + 1)

        load_dicom(im)

        counter += 1


def prepare_dataset(patient_df, img_dir, train=False):
    """
    prepare a tensorflow dataset for optimal operations
    in batches allowing for big dataset sizes

    :param patient_df: pd.DataFrame
    :param img_dir: str
    :param train: boolean
    :return: tf.data.Dataset
    """
    if train:
        dataset = load_dataset(patient_df, img_dir=img_dir, fvc_col='FVC')
    else:
        dataset = load_dataset(patient_df, img_dir=img_dir)

    # TODO can replace parallel calls w/ AUTOTUNE
    dataset = dataset.map(parse_image, num_parallel_calls=4)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(BUFFER_SIZE)

    return dataset


def load_dataset(patient_df, img_dir, features=['FVC', 'Age', 'Patient'], fvc_col=None):
    """
    loads image + label (optional) data
    :param patient_df: pd.DataFrame
    :param features: list
    :param img_dir: str
    :param fvc_col: str
    :return: tf.data.Dataset
    """
    patient_data = patient_df[features].copy()
    patient = patient_data.pop('Patient')

    if fvc_col:
        target = patient_data.pop(fvc_col)
        dataset = tf.data.Dataset.from_tensor_slices((patient, (patient_data, target)))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((patient, patient_data))

    return dataset


def parse_image(filename, meta):
    """
    reads DICOM image and resizes to bring all to
    common size for training and inference
    :param filename: str
    :return: img: tf.Tensor
    """
    image_bytes = tf.io.read_file(filename)
    image = tfio.image.decode_dicom_image(image_bytes, on_error='strict', dtype=tf.float32)

    image = tf.image.resize(image, IMG_RESIZE)
    image = tf.reshape(image, image.shape[1:3])
    return image, meta

