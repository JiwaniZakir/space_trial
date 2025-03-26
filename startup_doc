
Name: TBD spaces 


User Journey:

User login the app/ creates account (puts username, password, and address)
Then user can take a picture or upload from camera, v2 video
User puts dimensions (estimate) and their current location (for services)
User chooses style via predefined style eg (mordern, scandinavian, mid century) or puts prompt. Users also purpose of design and products (optmization of space, redesigning, design from scratch, or etc)(they don’t have to choose a style, choose a style for you with optionality). 
Parameters put in the backend and image generation in accordance to their prereqs. (other options you can explore).
User can rearrange products with realistic dimensions in their image
Option to scroll through alternate products (in filters of store, availability, pricing)
User chooses what products to buy from image and shopping cart is created.
User checkout adds billing details
User receives products.
Personalization of styles and recommendations based on previous searches. 
User images and also create a workspace and upgrade and build it overtime. 


Could be refurbished products ie fb marketplace


TODO:
Market research 
Video launch
Eng doc with eng requirements 


https://github.com/sophiachann/ObjectDetectionProject-IKEAFurnituresRecommender
















2. Backend Development Plan
Phase 1: Image Recognition and CBIR
Steps:
Image Recognition API:


Integrate a pre-trained model or API like Google Vision or AWS Rekognition.
Test object detection and classification on user-uploaded images.
CBIR Implementation:


Use OpenCV, FAISS, or Milvus to compare inspiration images with product images.
Create a basic pipeline:
Extract features from images (e.g., color, shape, texture).
Match features with the product database.
Testing:


Test with a small sample dataset of images and verify accuracy.






Object similarity model compares existing products on different product database essentially scrapping the web for similar products and suggestions. 

Creates a affiliate link on eligible websites i.e amazon

Swap, compare products based on provider and price and user can visualize it 

Product can be clicked and added to cart
Phase 1: Object Detection & Segmentation
Goal: Accurately detect and extract furniture from user-uploaded images so users can swap them out.
Step 1: Choose an Object Detection & Segmentation Model
✅ Use Detectron2 (Mask R-CNN) for pixel-perfect cutouts
Detects and isolates furniture in images at a pixel level


Enables swapping furniture realistically


✅ Alternative models to consider
Segment Anything Model (SAM): Good for general segmentation, but may require fine-tuning


YOLOv8 + SAM Hybrid: YOLO for detection, SAM for fine-grained segmentation



Step 2: Train the Object Detection Model
✅ Dataset Collection
Use existing datasets like ADE20K, OpenImages, COCO Furniture, and IKEA datasets


If necessary, create a custom dataset by scraping furniture images with segmentation masks


✅ Preprocessing the Data
Resize images (512x512 or 1024x1024)


Augment data (flip, rotate, color adjust)


Label images using tools like LabelMe or CVAT


✅ Fine-Tune Detectron2 on Furniture Data
python
CopyEdit
# Install Detectron2
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# Import dependencies
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("custom_furniture_train",)
cfg.DATASETS.TEST = ("custom_furniture_test",)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20  # Number of furniture categories
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/model_final.pth"

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

✅ Deploy the Model via API
Convert to ONNX/TensorRT for faster inference


Deploy using FastAPI or Flask



🟡 Phase 2: Image Generation (AI-Based Inspiration)
Goal: Generate an AI-powered design inspiration based on user preferences.
Step 3: Choose an Image Generation Model
✅ Use Stable Diffusion (ControlNet) for Interior Design
Allows users to generate realistic room inspirations


Input: User-selected style, existing room image


Output: AI-generated inspiration image


✅ Fine-Tune Stable Diffusion for Interior Styles
Train on interior design datasets (Pinterest, ArchDaily, Houzz, etc.)


Implement ControlNet for structure preservation


✅ Example Workflow
User uploads an image of their room


User selects a style (Scandinavian, Modern, etc.)


Stable Diffusion generates an AI-enhanced version



🟠 Phase 3: Product Matching & Web Scraping
Goal: Match AI-generated inspiration with real products that users can swap and buy.
Step 4: Web Scraping for Product Database
✅ Scrape furniture sites (Amazon, Wayfair, IKEA, Walmart)
Use Scrapy / BeautifulSoup for static content


Use Selenium / Playwright for dynamic content


Store image URLs, prices, descriptions, and availability in a MongoDB/PostgreSQL database


✅ Automate Periodic Updates
Run scrapers weekly to keep data fresh



Step 5: Image-Based Product Matching
✅ Extract Features from Product Images
Use ResNet50 / ViT (Vision Transformer) to encode furniture images into vectors


✅ Store Image Features in FAISS for Fast Search
FAISS (Facebook AI Similarity Search) enables quick nearest-neighbor search


✅ Search Flow
Convert the AI-generated inspiration into feature vectors


Find similar products in the database using FAISS/Milvus


Filter results by price, brand, availability


python
CopyEdit
import faiss
import numpy as np

# Load extracted image features
product_vectors = np.load("product_features.npy")

# Initialize FAISS index
index = faiss.IndexFlatL2(product_vectors.shape[1])
index.add(product_vectors)

# Search for nearest product match
D, I = index.search(user_inspiration_vector, k=5)  # Top 5 matches

✅ Display Results
Show visually similar furniture that can be swapped into the room


Allow filtering by price, brand, store availability



🔵 Phase 4: User Experience & Integration
Step 6: Implement User Interaction & Swapping
✅ Drag-and-Drop UI for Swapping
Allow users to replace detected objects with matched products


Use three.js or WebGL for 3D visualization


✅ Add "Try in My Room" Feature (AR/VR)
Use WebAR (for browsers) or ARKit/ARCore (for mobile apps)


✅ Checkout & Affiliate Integration
Generate affiliate links for Amazon, Wayfair, or other stores



🚀 Final Deployment & Scaling
✅ Deploy AI Models in Production
Host models on AWS Lambda / Google Cloud AI


Optimize inference with TensorRT / ONNX


✅ Monitor Performance & Improve UX
Track user behavior to refine AI recommendations



