#!/bin/bash
gcloud auth activate-service-account  --key-file=sinuous-client-274919-bb1d7df46d87.json
for var in "$@"
do
    gsutil cp "$var" gs://dl2020
done