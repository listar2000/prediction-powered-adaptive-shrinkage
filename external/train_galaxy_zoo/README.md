# Galaxy Zoo ResNet50 training and prediction

These scripts lay out the data preparation, training, and inference pipeline
used to produce the Galaxy Zoo predictions in the PAS experiments. They are a
light organization of the original research code, not a newly validated
training package.

## External data layout

The raw images and metadata are intentionally not committed. Download the
Galaxy Zoo bundle separately and arrange it as follows:

```text
external/assets/galaxy_zoo/
├── images/
│   ├── <galaxyID>.jpg
│   └── ...
└── metadata/
    └── source/
        ├── gz2_train.csv
        ├── gz2_valid.csv
        ├── gz2_test.csv
        ├── gz2_filename_mapping.csv
        └── gz2sample.csv
```

The three `gz2_{train,valid,test}.csv` files are the pre-labeled eight-class
splits consumed by the retained pipeline and must include `galaxyID` and
`label1`. The filename mapping must contain `asset_id` and `objid`, and
`gz2sample.csv` must contain `OBJID` and `WVT_BIN`. The large Hart et al.
classification table (`gz2_hart16.csv`) is source provenance for these labels
but is not consumed directly by the retained scripts; it may remain in the
download bundle rather than the Git repository.

All paths in `config.yaml` are relative to `external/assets/galaxy_zoo/`.
That entire assets directory is ignored by Git, including model checkpoints
and generated prediction files.
