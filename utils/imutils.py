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


def prepare_dataset(dirname, train=False, label_dir=None):
    """"""

    if train:
        dataset = load_dataset(dirname, fvc_dir=label_dir)
    else:
        dataset = load_dataset(dirname)

    # TODO can replace parallel calls w/ AUTOTUNE
    dataset = dataset.map(parse_image, num_parallel_calls=4)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    dataset = dataset.prefetch()

    return dataset


def load_dataset(img_dir, fvc_dir=None):
    """"""
    filenames = tf.io.gfile.glob(img_dir + '/*/*.dcm')
    dataset = tf.data.Dataset.from_tensor_slices((filenames))

    if fvc_dir:
        dataset.map()
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    else:
        dataset = tf.data.Dataset.from_tensor_slices((filenames))

    return dataset


def parse_image(filename):
    """"""
    dataset = pydicom.dcmread(filename)
    img = dataset.pixel_array

    img = tf.image.resize_images(img, IMG_RESIZE)

    return img


def get_labels(label_path, filename):
    """"""
    df = pd.read_csv(label_path)
    patient_data = df[df['Patient'] == filename]
    return patient_data['FVC']