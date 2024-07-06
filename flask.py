from flask import Flask, request, jsonify
from transformers import pipeline
import os
app = Flask(__name__)

phrase_map = {
    'Alhamdulillah': "الحمد لله",
  'Good bye': "مع السلامة",
  'Good evening': "مساء الخير",
  'Good morning': "صباح الخير",
  'How are you': "ايه الاخبار",
  'I am pleased to meet you': "فرصة سعيدة",
  'I_m fine': "انا كويس",
  'I_m sorry': "انا اسف",
  'Not bad': "مش وحش ",
  'Salam aleikum': "السلام عليكم",
  'Sorry (Excuse me)': "لو سمحت",
  'Thanks': "شكرا"
}
video_cls = pipeline(model="mohamedsaeed823/VideoMAEF-finetuned-ARSL-diverse-dataset")

@app.route('/classify_video', methods=['POST'])
def classify_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    
    # Save the video file to a temporary location
    video_path = "video.mp4"
    video_file.save(video_path)
    
    # Perform video classification
    try:
        result=video_cls(video_path,top_k=1,frame_sampling_rate=6) # try to sample a frame every 6 seconds for better video understanding if the video is long enough
    except Exception as e:
        result=video_cls(video_path,top_k=1,frame_sampling_rate=3) # if the video is not long enough sample every 3 seconds

    # Extract the top label from the classification results
    top_label = result[0]['label']
    
    return jsonify({'result': phrase_map[top_label],'score': result[0]["score"]})

if __name__ == "__main__":
    app.run(debug=True)
