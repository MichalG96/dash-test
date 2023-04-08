# Tasks
### 1. Dumping data into parquet
- Rename existing parquet file (`data.parquet`) to `data_backup.parquet`
- Read `data_backup.parquet` file
- Read `data_current.csv`
- merge those files into one
- save merged df as `data.parquet`
### 2. Creating new CSV file
- Create skeleton file for 30 days
- Save it as `data_current.csv`

# Script usage

```shell
python utils.py migrate --start-date=2023-04-07 --tod=evening
```
