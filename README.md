# ObjectDetectron

ObjectDetectron is a Python repository for DETR-based object detection and visualization.

## Setup Environment (Linux)

### 1. Clone the Repository

```bash
git clone git@github.com:BrandonMagana/ObjectDetective.git 
cd ObjectDetective
```
### 2. Create and Activate Virtual Environment

```bash
# Install virtualenv if not already installed
sudo apt install python3-venv

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Code
You can now run the main script to perform object detection on an image.  
Replace `path/to/your/image.jpg` with the path to your image file.

```
python3 main.py -image path/to/your/image.jpg
```
The detected objects will be highlighted in the displayed image.

### Optional: Deactivate Virtual Environment
Once you are done, you can deactivate the virtual environment.


```bash
deactivate
```
