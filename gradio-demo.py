import gradio as gr
from transformers import pipeline

video_cls = pipeline(model="mohamedsaeed823/VideoMAEF-finetuned-ARSL-diverse-dataset")
phrase_map = {
    'Alhamdulillah': "الحمد لله",
  'Good bye': "مع السلامة",
  'Good evening': "مساء الخير",
  'Good morning': "صباح الخير",
  'How are you': "ايه الاخبار",
  'I am pleased to meet you': "فرصة سعيدة",
  'I am fine': "انا كويس",
  'I am sorry': "انا اسف",
  'Not bad': "مش وحش ",
  'Salam aleikum': "السلام عليكم",
  'Sorry': "لو سمحت",
  'Thanks': "شكرا"
}
def classify_video(video_path):
    try:
        result=video_cls(video_path,top_k=3,frame_sampling_rate=6) # try to sample a frame every 6 seconds for better video understanding if the video is long enough
    except Exception as e:
        result=video_cls(video_path,top_k=3,frame_sampling_rate=3) # if the video is not long enough sample every 3 seconds

    # Extract the top 3 label and their scores from the classification results
    top_label = [phrase_map[result[0]['label']], phrase_map[result[1]['label']], phrase_map[result[2]['label']]]
    top_label_confidence = [result[0]['score'], result[1]['score'], result[2]['score']]
    return dict(zip(top_label, top_label_confidence))

title = "Arabic Sign Language Recognition using VideoMAE"

examples = ["examples/alhamdulellah.mp4",
            "examples/forsa sa3eda.mp4",
            "examples/ma3a el salama.mp4",]
demo = gr.Interface(fn=classify_video,title=title, inputs=gr.Video(), outputs=gr.Label(num_top_classes=3), examples=examples)

if __name__ == "__main__":
    demo.launch()