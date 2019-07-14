# DeepBlue AOD Retrieval

At many wavelengths of visible light, the contrast between aerosols and the surface is difficult to discern, but in the 412 nm band——the "deep blue" band, aerosol signals tend to be bright and surface features dark.

## Command:
> Example:

```shell
python aod_retrieval_db.py --goci ../aod_retrieval_data/L1B1_land/L1B1_land-14-1.tif --solz ../aod_retrieval_data/SOLZ_land/SOLZ_land-14-1.tif --myd09 ../aod_retrieval_data/MYD09/MYD09A1-project1.tif --cloud ../aod_retrieval_data/cloud_land/cloud_land-14-1.tif --lut ../aod_retrieval_data/LUT/modis_lut_m.txt --output ../aod_retrieval_res/
```