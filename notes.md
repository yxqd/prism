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

# 20260225
* run sharded workflow
  `$ python scripts/run_workflow.py s3://prism-processed/sharded/tiny-imagenet-200/train/ --sharded`

This I/O peaks come from writing shards to s3. Should try making it smoother with overlapping of I/O and compute.
<img width="1904" height="968" alt="image" src="https://github.com/user-attachments/assets/3f729c58-1442-460b-a79b-0dfa14f61eb5" />
