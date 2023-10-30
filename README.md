# ID Card Scanner and Face Recognition
## Setting up environment
```
conda env create -f environment.yml
conda activate CVnApp
```
If you want to run the project with GPU, install onnxruntime-gpu
```
pip install onnxruntime-gpu
```
## Preparing model and database
1. Create folder database/face_database and weights
2. Download the face_database and weights files from https://github.com/laichithien/ID_Card_Scanner_Face_Recognition/releases
3. Organize your folders and files structure as below
```
    .
    ├── ...
    ├── database
    │   ├── face_database
    │   │   ├── name1.npy
    │   │   ├── name2.npy
    │   │   └── ...
    │   └── ...
    ├── weights
    │   ├── weights1.onnx
    │   ├── weights2.onnx
    │   └── ...
    └── ...
```