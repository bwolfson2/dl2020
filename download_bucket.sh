#!/bin/bash
gcloud auth activate-service-account  --key-file=client_secrets.json
gsutil cp "gs://dl20202/$1" $2 