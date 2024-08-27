import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr

# Set up the BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate caption for a single frame
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt", padding=True)
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Function to process video and generate captions
def process_video(video_path, frames_per_second=1):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps // frames_per_second
    
    frame_captions = {}
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            caption = generate_caption(rgb_frame)
            frame_captions[frame_count] = caption
        
        frame_count += 1
    
    cap.release()
    return frame_captions

# Gradio interface function
def gradio_process_video(video_file):
    captions = process_video(video_file.name)
    return "\n".join([f"Frame {k}: {v}" for k, v in captions.items()])

# Create Gradio interface
iface = gr.Interface(
    fn=gradio_process_video,
    inputs=gr.File(label="Upload Video"),
    outputs=gr.Textbox(label="Video Captions"),
    title="Video Frame Captioning",
    description="Upload a video to generate captions for its frames."
)

# Launch the interface
iface.launch()