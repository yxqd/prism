# 20260223
* created s3 buckets prism-landing/processed/model-collection (...-model-collection enabled versioning)
* added download_imagenet.py and use it to download tiny-imagenet-200
* use aws sync to upload to s3:prsim-landing/tiny-imagenet-200 (partially)
* added ingest.py script and use it to inject metadata for a portion of images:
  `$ python scripts/ingest.py s3://prism-landing/tiny-imagenet-200/train/n01443537/images --upload-metadata`

# 20260224
* added workflow script that can utilize dask
  `$ python scripts/run_workflow.py s3://prism-landing/tiny-imagenet-200/train/n01443537/images/`
* added sharding script
  `$ python scripts/shard_dataset.py --config config/shard.example.yaml`
