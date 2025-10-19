
# NautData
We construct **NautData**, a large-scale underwater instruction-following dataset containing 1.45 M image-text pairs, enabling developments and evaluations of underwater LMMs. 

<p align="center">
    <img src="../Figs/Table1.png" width="90%"/>
<p> 

## Key features
- Extensive underwater information encompassing image-, region-, and object-level understanding.
- Diverse conversational structures, including both rule-based and free-form content.
- Significant scale, comprising $158$ K images and $1.45$ M QA pairs.

## Distribution of NautData
<p align="center">
    <img src="../Figs/Distribution.png" width="90%"/>
<p>  

## Data engine
<p align="center">
    <img src="../Figs/Dataengine.png" width="90%"/>
<p> 

## Task and chat Data format
NautData includes eight tasks covering understanding at three granularities: image-, region-, and object-levels.  
### Image caption & VQA data
```
  {
    "id": "0",
    "image": "path/to/image.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nuser input" # Image caption prompt/visual question
      },
      {
        "from": "gpt",
        "value": "assistant output" # Image caption/visual answer
      }
    ]
  }
```
### Region caption data
```
  {
    "id": "1",
    "image": "path/to/image.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nPlease give a brief description of the region [bbox] in this underwater image." # [bbox] is pixel coordinate from dataset annotations 
      },
      {
        "from": "gpt",
        "value": "Region caption" # Data sourced from multiple LMMs
      }
    ]
  },
```
### Grounding data
```
  {
    "id": "2",
    "image": "path/to/image.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nPlease locate the bounding box coordinates of the region [Region caption]" 
      },
      {
        "from": "gpt",
        "value": "bbox" 
      }
    ]
  },
```
### Detection data
```
  {
    "id": "3",
    "image": "path/to/image.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nDetect all underwater object in the image, including class1, class2, ..., classN. Provide the category and regression box for each detected object." 
      },
      {
        "from": "gpt",
        "value": "label1:bbox1\nlabel2:bbox2\n...\nlabelM:bboxM" 
      }
    ]
  },
```
### Refering object classification data
```
  {
    "id": "4",
    "image": "path/to/image.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nClassify the object located inside the provided regression box [bbox]. Choose one of the following categories as the answer:class1, class2, ..., classN." 
      },
      {
        "from": "gpt",
        "value": "label" 
      }
    ]
  },
```
### Fine fish classification data
```
  {
    "id": "5",
    "image": "path/to/image.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat is the biological class of the fish in the image? Choose one of the following categories as the answer::class1, class2, ..., classN." 
      },
      {
        "from": "gpt",
        "value": "label" 
      }
    ]
  },
```
### Counting data
Choice data:
```
  {
    "id": "6",
    "image": "path/to/image.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nFind out how many fish are in the image and select the appropriate choice from the options below.\nA:N1\nB:N2\nC:N3\nD:N4\n" # The interval between options is selected from [5, 50, 100], and the order of the options will be shuffled.
      },
      {
        "from": "gpt",
        "value": "option"
      }
    ]
  },
```
Regression data:
```
  {
    "id": " 7",
    "image": "path/to/image.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nPlease count how many fish are in this image."
      },
      {
        "from": "gpt",
        "value": "num"
      }
    ]
  },
```
## Directory Structure of the Dataset
```
├──Images
    ├──aquarium_UOD
      ├── ...
    ├──Brackish
      ├── ...
    ├──Deepfish
      ├── ...
    ├──Fishnet
      ├── ...
    ├──IOCfish5k
      ├── ...
    ├──RUOD
      ├── ...
    ├──UIIS
      ├── ...
    ├──Underwater_garbage
      ├── ...
    ├──USIS10k
      ├── ...
    └──UVOT-400
      ├── ...
```
Please note that certain datasets must be renamed in accordance with the specified names above. For detailed instructions on how to download each dataset and their respective directory structures, please refer to the comprehensive guidelines provided in Nautdata.  
We also provide a subset of images from each dataset to facilitate a quick understanding of our **NautData**.

## Visualization

<p align="center">
  <img src="../Figs/Visualization.png" width="400"/>
<p>




