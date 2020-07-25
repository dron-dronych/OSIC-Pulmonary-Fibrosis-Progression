import random
import pydicom


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


def prepare_dataset(dirname, train=False):
    """"""
    dataset = load_dataset(dirname)

    if train:
        pass

    # TODO can replace parallel calls w/ AUTOTUNE
    dataset = dataset.map(parse_image, num_parallel_calls=4)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    dataset = dataset.prefetch()

    return dataset


def load_dataset(img_dir, fvc_dir=None):
    """"""
    filenames = tf.io.gfile.glob(img_dir + '/*/*.dcm')

    if fvc_dir:
        labels = get_labels(fvc_dir)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    else:
        dataset = tf.data.Dataset.from_tensor_slices((filenames))

    return dataset


def parse_image(filename):
    """"""
    dataset = pydicom.dcmread(img_path)
    img = dataset.pixel_array

    img = tf.image.resize_images(img, IMG_RESIZE)

    return img