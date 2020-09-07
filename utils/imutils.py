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
    """"""
    if train:
        dataset = load_dataset(patient_df, img_dir=img_dir, fvc_col='FVC')
    else:
        dataset = load_dataset(patient_df, img_dir=img_dir)

    # TODO can replace parallel calls w/ AUTOTUNE
    #     dataset = dataset.map(parse_image, num_parallel_calls=4)
    #     dataset = dataset.batch(BATCH_SIZE) # do we need batching with variable # images per scan???
    dataset = dataset.repeat()
    dataset = dataset.prefetch(BUFFER_SIZE)

    return dataset


def load_dataset(patient_df, img_dir, fvc_col=None):
    """"""
    cols = ['FVC', 'Age', 'Patient']
    patient_data = patient_df[cols].copy()
    patient = patient_data.pop('Patient')
    patient = patient.apply(lambda x: img_dir + x)

    dataset = tf.data.Dataset.from_tensor_slices(patient)
    dataset = dataset.map(load_images)

    if fvc_col:
        target = patient_data.pop(fvc_col)

        meta_dataset = tf.data.Dataset.from_tensor_slices((patient_data, target))
        #         dataset = dataset.concatenate(meta_dataset)
        dataset = tf.data.Dataset.zip((dataset, meta_dataset))
    else:
        meta_dataset = tf.data.Dataset.from_tensor_slices(patient_data)
        dataset = tf.data.Dataset.zip((dataset, meta_dataset))
    #         dataset = dataset.concatenate(meta_dataset)

    return dataset


def load_images(patient):
    #     filenames = tf.py_function(glob.glob, [patient + '/*.dcm'], Tout=tf.int64)
    filenames = tf.io.matching_files(patient + '/*.dcm')
    #     images = tf.py_function(pydicom.dcmread, [filenames], Tout=tf.int64)
    #     images = tf.map_fn(lambda x: pydicom.dcmread(x, filenames)
    image_bytes = tf.map_fn(tf.io.read_file, filenames, dtype=tf.string)
    images = tf.map_fn(lambda x: tfio.image.decode_dicom_image(x, on_error='strict', dtype=tf.float32), image_bytes,
                       dtype=tf.float32)
    images = tf.map_fn(lambda x: tf.image.resize(x, IMG_RESIZE), images)

    #     try:
    #         images.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    #     except AttributeError:
    #         warnings.warn(f'Patient {images[0].PatientID} CT scan does not '
    #                       f'have "ImagePositionPatient". Assuming filenames '
    #                       f'in the right scan order.')

    #     image = np.stack([s.pixel_array.astype(float) for s in images])
    #     try:
    #         image = tf.math.reduce_sum(images, axis=0)
    #     except RuntimeError as e:
    #         image = np.nan

    return images