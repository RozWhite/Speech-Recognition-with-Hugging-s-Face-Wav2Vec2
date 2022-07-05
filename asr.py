import torch
import speech_recognition as sr
import io
from pydub import AudioSegment
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Loading fine-tuned model and processor
tokenizer = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")
model = Wav2Vec2ForCTC.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")
recognizer = sr.Recognizer()

# Obtain audio from the microphone
with sr.Microphone(sample_rate=16000) as source:
    print("Please speak now")
    while True:
        audio=recognizer.listen(source)
        data = io.BytesIO(audio.get_wav_data())
        clip = AudioSegment.from_file(data)
        tensor = torch.FloatTensor(clip.get_array_of_samples()) # Convert array to tensor
        
        inputs = tokenizer(tensor, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
        logits = model(inputs).logits # Forward it to the model
        tokens = torch.argmax(logits, dim=-1) # Decode it
        text = tokenizer.batch_decode(tokens)
        print("You said: ", str(text).lower())
    

    
    
    

    
    
    
    

    