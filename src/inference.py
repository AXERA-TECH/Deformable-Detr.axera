import argparse
import numpy as np
import sys
import os

try:
    import axengine as ort
    print("Running on AXera NPU (axengine)...")
except ImportError:
    import onnxruntime as ort
    print("Running on CPU/GPU (onnxruntime)...")

from PIL import Image, ImageDraw, ImageFont

NORMALIZATION_ENABLED = False
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)

CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic_light",
    "fire_hydrant", "stop_sign", "parking_meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports_ball", "kite", "baseball_bat", "baseball_glove", "skateboard", "surfboard",
    "tennis_racket", "bottle", "wine_glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot_dog", "pizza", "donut", "cake", "chair", "couch",
    "potted_plant", "bed", "dining_table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell_phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy_bear", "hair_drier", "toothbrush"
]

def preprocess_normalized(image_path, input_h, input_w, layout="NCHW"):
    raw_image = Image.open(image_path).convert("RGB")
    img_w, img_h = raw_image.size
    
    scale = min(input_w / img_w, input_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    resized_image = raw_image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (input_w, input_h), (0, 0, 0)) 
    canvas.paste(resized_image, (0, 0))
    image_data = np.array(canvas, dtype=np.float32)

    if NORMALIZATION_ENABLED:
        image_data = (image_data - MEAN) / STD

    if layout == "NCHW":
        image_data = image_data.transpose(2, 0, 1)
    
    image_data = np.expand_dims(image_data, 0)
    return image_data, raw_image, {"original_size": (img_w, img_h), "scale": scale}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--output", type=str, default="result.jpg")
    parser.add_argument("--thresh", type=float, default=0.3)
    opt = parser.parse_args()

    session = ort.InferenceSession(opt.model)
    input_meta = session.get_inputs()[0]
    
    if input_meta.shape[1] == 3:
        layout, h, w = "NCHW", input_meta.shape[2], input_meta.shape[3]
    else:
        layout, h, w = "NHWC", input_meta.shape[1], input_meta.shape[2]

    img_tensor, raw_img, meta = preprocess_normalized(opt.img, h, w, layout)
    outputs = session.run(None, {input_meta.name: img_tensor})

    dets = outputs[0][0]    
    labels = outputs[1][0]  
    scores = dets[:, 4]
    keep = scores >= opt.thresh
    
    v_dets = dets[keep]
    v_labels = labels[keep]

    orig_w, orig_h = meta["original_size"]
    scale = meta["scale"]

    print(f"Detected {len(v_dets)} objects.")

    if len(v_dets) > 0:
        draw = ImageDraw.Draw(raw_img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except:
            font = ImageFont.load_default()

        for i in range(len(v_dets)):
            box = v_dets[i, :4] / scale
            score = v_dets[i, 4]
            label_id = int(v_labels[i])
            
            x1, y1, x2, y2 = box
            x1, x2 = np.clip([x1, x2], 0, orig_w)
            y1, y2 = np.clip([y1, y2], 0, orig_h)

            draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
            
            name = CLASSES[label_id] if label_id < len(CLASSES) else f"obj_{label_id}"
            text = f"{name} {score:.2f}"
            
            draw.rectangle([x1, y1-20, x1+100, y1], fill="lime")
            draw.text((x1+2, y1-20), text, fill="black", font=font)

    raw_img.save(opt.output)
    print(f"Result saved to {opt.output}")

if __name__ == "__main__":
    main()