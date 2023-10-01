# F-ViT: Build Open-Vocabulary Object Detectors Upon Frozen CLIP ViTs
## Requirements
The detection framework is built upon MMDetection2.x. To install MMDetection2.x, run

```bash
cd ~/your/project/directory
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.7.0
MMCV_WITH_OPS=1 pip install -e . -v
cd ..
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.28.1
pip install -e . -v
```
For other installation methods, please refer to the official website of 
[MMCV](https://github.com/open-mmlab/mmcv.git) and [MMDetection](https://github.com/open-mmlab/mmdetection.git).

## Data

## Checkpoints

## Test 
We provide the checkpoints of the object detectors.

## Train
We provide the checkpoints of EVA-CLIP ViTs that are refined by CLIPSelf.
