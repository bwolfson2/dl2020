#!/bin/bash
gcloud auth activate-service-account  --key-file=client_secrets.json
for var in "$@"
do
    gsutil cp "$var" gs://dl20202
done