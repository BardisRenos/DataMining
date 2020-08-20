# DataMining

<p align="justify">
Data mining is a process of discovering patterns in large data sets involving methods at the intersection of machine learning, statistics, and database systems. Data mining is an interdisciplinary subfield of computer science and statistics with an overall goal to extract information (with intelligent methods) from a data set and transform the information into a comprehensible structure for further use.
</p>

<p align="center"> 
<img src="https://github.com/BardisRenos/DataMining/blob/master/data-mining.jpg" width="350" height="200" style=centerme>
</p>

### The project

<p align="center">
"Edge Prediction on the Greek Web"
</p>
  
<p align="justify">
The main purpose of this work is to apply data extraction methods to predict missing edges of a graph. For this purpose we will apply mining techniques for text and for the graph. For the implementation of the work I used the version 3.5 of Python for the reason of text encoding.
</p>


<p align="justify">
Since each folder contains more than one txt file, for this check if there are duplicate files (text files in the same nodes) and delete them to reduce the volume that I will edit in the text field. Having a smaller number of text files will help reduce editing time and better editing and better results.
</p>

```python
import os
import zipfile
from shutil import rmtree
path = "/home/renos/Desktop/dataset.zip"
dir_name = "/home/renos/Desktop/dataset"
dir_source = "/home/renos/Desktop/dataset/hosts"

dir_destination = "/home/renos/Desktop/hosts"

extension = ".zip"
zipfile.ZipFile(path).extractall(dir_name)
os.chdir(dir_source)

for onoma in os.listdir(dir_source):
    if onoma.endswith(extension):
        file_name = os.path.abspath(onoma)
        file_onoma = os.path.splitext(onoma)[0]
        # print(file_onoma)
        dir_namefiles = dir_destination+"/"+file_onoma
        os.makedirs(dir_namefiles)
        zip_ref = zipfile.ZipFile(file_name)
        zip_ref.extractall(dir_destination+"/"+file_onoma)

for root, dirs, files in os.walk(dir_name):
  rmtree(root)
```


