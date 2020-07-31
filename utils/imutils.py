import random
import pydicom
import tensorflow as tf


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
    """"""
    dataset = load_dataset(dirname, fvc_dir=fvc_dir, train_df=train_df)
    #     dataset = dataset.map(lambda x: tf.py_function(
    #         func=tensor_to_string, inp=[x], Tout=tf.string
    #     ))

    if train:
        pass

    # TODO can replace parallel calls w/ AUTOTUNE
    dataset = dataset.map(parse_image, num_parallel_calls=4)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    dataset = dataset.prefetch()

    return dataset


def load_dataset(img_dir, fvc_dir=None, train_df=None):
    """"""
    filenames = tf.io.gfile.glob(str(img_dir + '*/*.dcm'))
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if fvc_dir and train_df:
        dataset = dataset.map(partial(get_label, train_df), num_parallel_calls=4)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    return dataset


def parse_image(filename):
    """"""
    image_bytes = tf.io.read_file(filename)
    image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)

    img = tf.image.resize(image, IMG_RESIZE)

    return img


def get_label(df, filename):
    """"""
    patient_data = df[df['Patient'] == filename]
    return patient_data['FVC']


def tensor_to_string(tensor):
    return tensor