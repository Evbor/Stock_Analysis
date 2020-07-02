#!/bin/bash

if [ -z "${A}" ] || [ -z "${Q}" ]
then
  echo "Environment variables A and Q must be set to the API keys for Alphavantage's and Quandl's services respectively in order to configure the pipeline."
else
  echo N | stockanalysis config -a "${A}" -q "${Q}"
  exec stockanalysis run-pipeline ${GPU:+"-g"} "$@"
fi
