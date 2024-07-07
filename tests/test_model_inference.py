import pytest
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

@pytest.fixture(scope="module")
def load_trained_model(train_model):
    config = train_model
    model_dir = config.MODEL_DIR / "trained-model"
    model = tf.keras.models.load_model(model_dir)
    return model, config

def test_model_inference(load_trained_model):
    model, config = load_trained_model
    print("Model loaded successfully for inference.")

    OUTPUT_IMAGE_FILE = str(config.MODEL_DIR / "prediction" / f"{config.OUTPUT_NAME}.TFRecord")
    print(f"OUTPUT_IMAGE_FILE: {OUTPUT_IMAGE_FILE}")

    image_files_list = ["./tests/test_data/images/image_2021.tfrecord.gz"]
    json_file = "./tests/test_data/images/image_2021.json"

    with open(json_file, encoding='utf-8') as jm: mixer = json.load(jm)

    # Get relevant info from the JSON mixer file.
    patch_width = mixer["patchDimensions"][0]
    patch_height = mixer["patchDimensions"][1]
    patches = mixer["totalPatches"]
    patch_dimensions_flat = [patch_width * patch_height, 1]

    def parse_image(example_proto):
        columns = [
            tf.io.FixedLenFeature(shape=patch_dimensions_flat, dtype=tf.float32) for k in config.FEATURES
        ]
        image_features_dict = dict(zip(config.FEATURES, columns))
        return tf.io.parse_single_example(example_proto, image_features_dict)

    # Create a dataset from the TFRecord file(s).
    image_dataset = tf.data.TFRecordDataset(image_files_list, compression_type="GZIP")
    image_dataset = image_dataset.map(parse_image, num_parallel_calls=5)

    # Break our long tensors into many little ones.
    image_dataset = image_dataset.flat_map(
        lambda features: tf.data.Dataset.from_tensor_slices(features)
    )

    # Turn the dictionary in each record into a tuple without a label.
    image_dataset = image_dataset.map(
        lambda data_dict: (tf.transpose(list(data_dict.values())), )
    )

    image_dataset = image_dataset.batch(patch_width * patch_height)

    predictions = model.predict(image_dataset, steps=patches, verbose=1)
    print(f"predictions shape: {predictions.shape}")

    # Create the target directory if it doesn't exist
    Path(OUTPUT_IMAGE_FILE).parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing predictions to {OUTPUT_IMAGE_FILE} ...")
    writer = tf.io.TFRecordWriter(OUTPUT_IMAGE_FILE)

    # Every patch-worth of predictions we"ll dump an example into the output
    # file with a single feature that holds our predictions. Since our predictions
    # are already in the order of the exported data, the patches we create here
    # will also be in the right order.
    patch = [[], [], [], [], [], []]

    cur_patch = 1

    for i, prediction in enumerate(predictions):
        patch[0].append(int(np.argmax(prediction)))
        patch[1].append(prediction[0][0])
        patch[2].append(prediction[0][1])
        patch[3].append(prediction[0][2])
        patch[4].append(prediction[0][3])
        patch[5].append(prediction[0][4])


        if i == 0:
            print(f"prediction.shape: {prediction.shape}")

        if (len(patch[0]) == patch_width * patch_height):

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                    "prediction": tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=patch[0])),
                    "cropland_etc": tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=patch[1])),
                    "rice": tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=patch[2])),
                    "forest": tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=patch[3])),
                    "urban": tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=patch[4])),
                    "others_etc": tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=patch[5])),
                    }
                )
            )

            # Write the example to the file and clear our patch array so it"s ready for
            # another batch of class ids
            writer.write(example.SerializeToString())
            patch = [[], [], [], [], [], []]
            cur_patch += 1

    writer.close()

    # assert output file exists
    assert Path(OUTPUT_IMAGE_FILE).exists(), "Inference did not return any results."
