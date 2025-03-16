### conda activate trafficvio 


# Project Setup and GitHub Configuration

## 1. Generate SSH Key (if not already generated)
```sh
ssh-keygen -t rsa -b 4096 -C "kushals1992003@gmail.com"
```
- Press **Enter** to save it in the default location (`~/.ssh/id_rsa`).
- Set a passphrase if desired (or press Enter to leave it empty).

## 2. Start SSH Agent and Add Key
```sh
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
```

## 3. Add SSH Key to GitHub
1. Copy the SSH key to the clipboard:
   ```sh
   clip < ~/.ssh/id_rsa.pub
   ```
2. Go to **GitHub > Settings > SSH and GPG keys**.
3. Click **New SSH Key**, paste the copied key, and click **Add SSH Key**.

## 4. Verify SSH Connection
```sh
ssh -T git@github.com
```
Expected output:
```sh
Hi Kushal19903! You've successfully authenticated, but GitHub does not provide shell access.
```

## 5. Clone the Repository
```sh
git clone git@github.com:Kushal19903/your-repository-name.git
```

## 6. Navigate to Project Directory
```sh
cd your-repository-name
```

## 7. Track and Commit Changes
```sh
git add .
git commit -m "Initial commit with project setup"
```

## 8. Push Changes to GitHub
```sh
git push origin master
```

## Notes
- Ensure your **SSH key is added to GitHub** before cloning

### Comprehensive Explanation of Traffic Violation Detection System

## 1. Root Directory Files

### 1.1. `detect.py`

This is the main script that performs object detection and violation detection.

**Purpose:** Processes input images/videos, detects objects, and identifies violations.

**Key Code Sections:**

```python
# Load YOLOv5 model
model = attempt_load(weights, map_location=device)  # load FP32 model

# Process frames
for path, img, im0s, vid_cap in dataset:
    # Inference
    pred = model(img)[0]
    
    # Apply NMS (Non-Maximum Suppression)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    
    # Process detections
    for i, det in enumerate(pred):
        # Extract bounding boxes, confidence scores, and class labels
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            
            # Check for violations (helmet, triple riding, license plate)
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                if names[c] == 'Motorcycle':
                    motorcycle_detected = True
                elif names[c] == 'Helmet':
                    helmet_detected = True
                elif names[c] == 'NumberPlate':
                    # Save number plate for OCR
                    save_one_box(xyxy, im0s, file=f'numberplate.jpg', BGR=True)
                    # Perform OCR
                    plate_number = ocr('numberplate.jpg')
                    # Validate license
                    validate_license(plate_number)
```

**How It Works:**

1. Loads the YOLOv5 model with trained weights
2. Processes each frame from the input source
3. Performs object detection to identify motorcycles, helmets, number plates, etc.
4. For each detection, applies logic to identify violations:
    - Checks if a motorcycle is detected without a helmet
    - Counts faces to detect triple riding
    - Extracts and validates license plates
5. Annotates the output frames with detection results
6. Sends alerts for detected violations

### 1.2. `openalpr_ocr.py`

This file contains the OCR functionality for reading text from number plates.

**Purpose:** Extracts text from license plate images.

**Key Code Sections:**

```python
def ocr(img_path):
    # OpenALPR API configuration
    SECRET_KEY = 'sk_fa2ed8536c60010729c3477d'
    url = 'https://api.openalpr.com/v2/recognize'
    
    # Prepare the image for OCR
    with open(img_path, 'rb') as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Send request to OpenALPR API
    r = requests.post(
        url,
        data=dict(
            image_base64=img_base64,
            recognize_vehicle=0,
            country='ind',
            secret_key=SECRET_KEY,
        )
    )
    
    # Process response
    try:
        result = r.json()
        if result['results']:
            return result['results'][0]['plate']
        else:
            return None
    except:
        return None
```

**How It Works:**

1. Takes an image path as input
2. Reads and encodes the image in base64 format
3. Sends the encoded image to the OpenALPR API
4. Processes the API response to extract the license plate text
5. Returns the recognized text or None if recognition fails

### 1.3. `best.pt`

This is the trained YOLOv5 model file.

**Purpose:** Contains the neural network weights for object detection.

**How It Works:**

- Stores the trained neural network architecture and weights
- Used by `detect.py` to identify objects in images/videos
- Trained to recognize specific classes: motorcycles, helmets, number plates, faces, etc.

### 1.4. `info.csv`

This is the database file for license plate information.

**Purpose:** Stores license plate numbers and their validity information.

**Structure:**

```plaintext
number,name,start,end
TN76AZ1197,aaaaaaaa,2/3/2020,27/3/2023
```

**How It Works:**

1. Used by the detection script to validate detected license plates
2. Contains columns for:
    - `number`: License plate number
    - `name`: Owner name
    - `start`: Start date of validity
    - `end`: End date of validity (expiration date)

### 1.5. `run_detection.bat`

A batch script for Windows to simplify running the detection script.

**Purpose:** Provides a convenient way to run the detection script with predefined parameters.

**Content Example:**

```plaintext
@echo off
python detect.py --weights best.pt --source 0 --conf 0.25 --view-img
```

## 2. Models Directory (`models/`)

This directory contains the model architecture definitions for YOLOv5.

### 2.1. `models/common.py`

Contains common building blocks for the YOLOv5 neural network.

**Purpose:** Defines reusable neural network components.

**Key Components:**

```python
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
```

**How It Works:**

1. Defines basic neural network building blocks like convolutions, bottlenecks, etc.
2. These components are used to construct the complete YOLOv5 architecture
3. Implements efficient operations for object detection

### 2.2. `models/yolo.py`

Defines the complete YOLOv5 model architecture.

**Purpose:** Implements the YOLO (You Only Look Once) neural network architecture.

**Key Components:**

```python
class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
        super(Model, self).__init__()
        # Load model configuration
        self.yaml = yaml.safe_load(open(cfg))
        self.model, self.save = parse_model(self.yaml, ch=[ch])  # model, save list
```

**How It Works:**

1. Loads the model configuration from a YAML file
2. Constructs the neural network based on the configuration
3. Implements the forward pass for inference

## 3. Utils Directory (`utils/`)

This directory contains utility functions for data processing, visualization, and other operations.

### 3.1. `utils/general.py`

Contains general utility functions used throughout the project.

**Purpose:** Provides helper functions for various operations.

```python
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """Runs Non-Maximum Suppression (NMS) on inference results"""
    pass
```

**How It Works:**

1. Implements Non-Maximum Suppression (NMS) to filter overlapping detections
2. Provides functions for coordinate transformations
