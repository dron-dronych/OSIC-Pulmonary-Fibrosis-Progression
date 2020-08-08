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


def prepare_dataset(dirname, train=False, label_dir=None, fvc_dir=None, train_df=None):
    """
    prepare a tensorflow dataset for optimal operations
    in batches allowing for big dataset sizes

    :param dirname: str
    :param train: boolean
    :param label_dir: str
    :param fvc_dir: str
    :param train_df: pd.DataFrame
    :return: tf.data.Dataset
    """
    dataset = load_dataset(dirname, fvc_dir=fvc_dir, train_df=train_df)

    # TODO can replace parallel calls w/ AUTOTUNE
    dataset = dataset.map(parse_image, num_parallel_calls=4)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(BUFFER_SIZE)

    return dataset


def load_dataset(img_dir, fvc_dir=None, train_df=None):
    """
    loads image + label (optional) data
    :param img_dir: str
    :param fvc_dir: str
    :param train_df: pd.Dataframe
    :return: tf.data.Dataset
    """
    filenames = tf.io.gfile.glob(img_dir + '*/*.dcm')
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if fvc_dir and train_df is not None:
        labels = dataset.map(partial(get_label, df=train_df), num_parallel_calls=4)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    return dataset


def parse_image(filename):
    """
    reads DICOM image and resizes to bring all to
    common size for training and inference
    :param filename: str
    :return: img: tf.Tensor
    """
    image_bytes = tf.io.read_file(filename)
    image = tfio.image.decode_dicom_image(image_bytes, on_error='strict',
                                          dtype=tf.float32)

    image = tf.image.resize(image, IMG_RESIZE)
    image = tf.reshape(image, image.shape[1:3])
    return image


def get_label(df, filename):
    """
    loads label data for each patient's CT scan
    :param df: pd.Dataframe
    :param filename: str
    :return: patient_fvc_data: pd.Series
    """
    df_ = df.copy()
    parts = tf.strings.split(filename, os.path.sep)
    patient_id = parts[-2]
    df_['Patient'] = df_['Patient'].apply(tf.constant)
    ids = []

    for i, j in df_['Patient'].iteritems():
        if tf.equal(j, patient_id):
            ids.append(i)

    patient_data = df_.loc[ids, 'FVC']
    return patient_data
