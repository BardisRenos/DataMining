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


### Data set
<p align="justify">
The data set is a large number .txt file which are stored into each folder with the folder name. That means each folder name represents a web node. Each text file has as data of the website text without the hyperlinks that may has. In that way I have to predict by text meaning the pair node of each given node.
</p>

<p align="justify">
Since each folder contains more than one text file, for this check if there are duplicate files (text files in the same nodes) and delete them to reduce the volume that I will edit in the text field. Having a smaller number of text files will help reduce editing time and better editing and better results.
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



<p align="justify">
In order to get some results we have to edit the text contained in the nodes and for this reason we have to "clen" the text and keep some words in which we can "extract" information. The texts contained in each node are grouped into a larger text for easier editing (after we have deleted the texts that are the same)
</p>

<p align="justify">
The texts that contain the nodes, their volume will be reduced because words with a short length, ie words with a length less than or equal to 4 are excluded. Next, we have a collection in a text file that contains words that have a high frequency and do not help in extracting information which are also called "Stopping Words". They must also have only words, ie symbols and numbers are excluded as well.
</p>

```python
import filecmp

lista = []
count = 0
listadiegrafis = []

metriris = 0
for root, directories, filenames in os.walk(dir_destination):
    lista.clear()
    for filename in filenames:
        diadromi = os.path.join(root, filename)
        ono = os.path.basename(diadromi)
        lista.append(diadromi)
    for w, j in zip(lista, lista[1::]):
        a = filecmp.cmp(w, j, shallow=True)
        if a:
            count += 1
            stixeio = os.path.basename(w)
            stoixeio1 = os.path.basename(j)
            listadiegrafis.append(j)

for i in listadiegrafis:
    os.remove(i)
    metriris += 1
print("Same text files:", metriris)
```
```text
Same text files:1798
```

### Word processing procedures

* Remove short words
* Remove words belonging to the Stopping Words collection
* Subtraction of numbers & symbols
* Remove punctuation

<p align="justify">
From the texts we create the table of tf-idf with the texts that have been edited and entered into a large list which contains lists of texts, in which each list is a text, ie each node is converted to large text (the which contains all text files if there is more than one).
Then we create the cosine similarity table from the tf-idf table which we have converted to a sparce matrix and store it in a numpy array so that we have it available when we do tests for the prediction model that we will apply later. At the same time we save a file in pickle format with the names of the nodes because there is no pickle file in the form of pickle with the names of the nodes because there is no tangible utility.
</p>

<p align="justify">
The size of the table should be 2041 lines on (the only words in the text) but the lines in our case, tf-idf has 2036 because 5 nodes the length of the text was too small or texts that contained words that did not had semantic value. Because lists of texts that are empty are excluded.
</p>

