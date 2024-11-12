---
title: 泰坦尼克
comments: true
---

## Jupyter Notebook

https://github.com/sigmax01/ml/blob/master/docs/project/kaggle/titanic.ipynb

### 自动从Kaggle下载数据

```python
# 导入数据集到相应位置, 如何设置密钥, 请见https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md
import zipfile
from pathlib import Path
from kaggle import KaggleApi
def check_and_download_dataset_from_kaggle(dataset_name):
    save_path = Path.home() / "kaggle_data" # 根据操作系统自动组合路径
    save_path.mkdir(parents=True, exist_ok=True) # 确保路径存在
    api = KaggleApi()
    api.authenticate()
    if "/" in dataset_name: # 根据是竞赛集还是用户集自动获取下载路径和函数
        username, dataset = dataset_name.split("/")
        zip_path = save_path / f"{dataset}.zip"
        extract_path = save_path / f"{dataset}"
        download_func = api.dataset_download_files
    else:
        zip_path = save_path / f"{dataset_name}.zip"
        extract_path = save_path / f"{dataset_name}"
        download_func = api.competition_download_files
    if extract_path.exists():
        print("文件早就下载好了ヾ(≧▽≦*)o")
        pass
    else:
        download_func(dataset_name, path=save_path)
        with zipfile.ZipFile(zip_path, "r") as f: # 解压下载的文件并删除
            f.extractall(extract_path)
        zip_path.unlink()
        print("下载完成(●'◡'●)")
    return extract_path
```