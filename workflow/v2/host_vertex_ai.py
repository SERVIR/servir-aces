# -*- coding: utf-8 -*-

import logging
logging.basicConfig(level=logging.INFO)

try:
    from aces.config import Config
except ModuleNotFoundError:
    logging.info("ModuleNotFoundError: Attempting to import from parent directory.")
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    from aces.config import Config

import os
import tensorflow as tf
import subprocess


# Load the trained model
# h5 won't work with vertex ai
# this_model = tf.keras.models.load_model(f"{str(Config.MODEL_DIR)}/{Config.MODEL_NAME}.h5")

if Config.USE_BEST_MODEL_FOR_INFERENCE:
    logging.info(f"Using best model for inference.\nLoading model from {str(Config.MODEL_DIR)}/{Config.MODEL_CHECKPOINT_NAME}.tf")
    this_model = tf.keras.models.load_model(f"{str(Config.MODEL_DIR)}/{Config.MODEL_CHECKPOINT_NAME}")
else:
    logging.info(f"Using last model for inference.\nLoading model from {str(Config.MODEL_DIR)}/{Config.MODEL_NAME}")
    this_model = tf.keras.models.load_model(f"{str(Config.MODEL_DIR)}/{Config.MODEL_NAME}")

logging.info(this_model.summary())

# Save the trained model in the google cloud storage bucket
gcs_save_dir = f"gs://{Config.GCS_BUCKET}/{Config.GCS_VERTEX_MODEL_SAVE_DIR}/{Config.MODEL_DIR_NAME}"
this_model.save(gcs_save_dir)
logging.info(f"Model saved to: {gcs_save_dir}")

def run_shell_process(exe):
    p = subprocess.Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while(True):
        # returns None while subprocess is running
        retcode = p.poll()
        line = p.stdout.readline()
        yield str(line)
        if retcode is not None:
            break

Config.MODEL_DIR_NAME = f"{Config.MODEL_DIR_NAME}_mt_{Config.GCP_MACHINE_TYPE.replace('-', '_')}"
Config.MODEL_DIR_NAME = "dnn_w_elv"

# Deploy
# delete model before deploying
try:
    delete_model = f"gcloud ai models delete {Config.MODEL_DIR_NAME} --project={Config.GCS_PROJECT} --region={Config.GCS_REGION}"
    logging.info(f"deleting model before deploying: {delete_model}")
    for line in run_shell_process(delete_model.split()):
        logging.info(line)
except Exception as e:
    logging.info(f"Exception in deleting model: {e}")

# upload model
upload_model = f"""gcloud ai models upload \
--artifact-uri={gcs_save_dir} \
--project={Config.GCS_PROJECT} \
--region={Config.GCS_REGION} \
--container-image-uri={Config.GCS_VERTEX_CONTAINER_IMAGE} \
--description={Config.MODEL_DIR_NAME} \
--display-name={Config.MODEL_DIR_NAME} \
--model-id={Config.MODEL_DIR_NAME}"""

logging.info(f"uploading model")
result = subprocess.check_output(upload_model, shell=True)
logging.info(f"{result}")

# deploy model

# create end_point
endpoint_create = f"""gcloud ai endpoints create \
--display-name={Config.MODEL_DIR_NAME + '_end_point'} \
--region={Config.GCS_REGION} \
--project={Config.GCS_PROJECT}"""

logging.info(f"creating end point")
result = subprocess.check_output(endpoint_create, shell=True)
logging.info(f"{result}")

# get end_point
endpoint_get = f"""gcloud ai endpoints list \
--project={Config.GCS_PROJECT} \
--region={Config.GCS_REGION} \
--filter=displayName:{Config.MODEL_DIR_NAME} \
--format="value(ENDPOINT_ID.scope())"\
"""
logging.info(f"getting endpoint")
result = subprocess.check_output(endpoint_get, shell=True)
logging.info(f"{str(result)}")

endpoints = result.decode("utf-8").split()
endpoint_id = endpoints[0]

# deploy the model
# deploy_model = f"""gcloud ai endpoints deploy-model {endpoint_id} \
# --project={Config.GCS_PROJECT} \
# --region={Config.GCS_REGION} \
# --model={Config.MODEL_DIR_NAME} \
# --display-name={Config.MODEL_DIR_NAME} \
# --machine-type={Config.GCP_MACHINE_TYPE}"""
# logging.info(f"deploying model:")
# result = subprocess.check_output(deploy_model, shell=True)
# logging.info(result)


deploy_model = f"""gcloud ai endpoints deploy-model {endpoint_id} \
--project={Config.GCS_PROJECT} \
--region={Config.GCS_REGION} \
--model={Config.MODEL_DIR_NAME} \
--display-name={Config.MODEL_DIR_NAME} \
--machine-type={Config.GCP_MACHINE_TYPE}"""
# --accelerator=type=nvidia-tesla-k80,count=1"""
# --deployed-model-id={Config.MODEL_DIR_NAME}"""
# --traffic-split=[{Config.MODEL_DIR_NAME}=100]"""
logging.info(f"deploying model:")
logging.info(deploy_model)
result = subprocess.check_output(deploy_model, shell=True)
logging.info(result)
