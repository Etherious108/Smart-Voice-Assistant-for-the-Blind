import os
import pickle
import numpy as np
import tensorflow as tf
# import tensorflow 
import json
import pandas as pd
from PIL import Image
# from tensorflow.python import *
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from django.shortcuts import render,redirect
from django.core.files.storage import FileSystemStorage
from .models import ImageCaption  
from .forms import ImageCaptionForm
from googletrans import Translator
from gtts import gTTS
from IPython.display import Audio

from django.http import JsonResponse

def landingpage(request):
    return render(request,'landingpage.html')

def generate_caption(image_path):
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the paths for 'features.pkl' and 'captions.txt'
    features_file = os.path.join(script_directory, 'features.pkl')
    captions_file = os.path.join(script_directory, 'captions.txt')

    # Load features from the saved pickle file
    with open(features_file, 'rb') as f:
        features = pickle.load(f)

    # Reading the descriptions.txt file
    with open(captions_file, 'r') as f:
        next(f)
        desc_doc = f.read()

    # Mapping the descriptions to the images
    mapping = {}
    for each_desc in desc_doc.split('\n'):
        tokens = each_desc.split(',')
        if len(each_desc) < 2:
            continue
        image_id, desc_of = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        desc_of = " ".join(desc_of)
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(desc_of)

    # Editing the descriptions: Convert to lower case and add beginning and ending
    def edit_description(mapping):
        for key, desc in mapping.items():
            for i in range(len(desc)):
                x = desc[i]
                x = x.lower()
                x = x.replace('[^A-Za-z]', '')
                x = x.replace('\s+', ' ')
                x = 'beginning ' + " ".join([word for word in x.split() if len(word) > 1]) + ' ending'
                desc[i] = x

    # Calling the preprocessing text function
    edit_description(mapping)

    # Appending all descriptions into a list: Each image with 5 descriptions
    img_desc = []
    for key in mapping:
        for caption in mapping[key]:
            img_desc.append(caption)

    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.models import load_model
    from keras.utils import to_categorical
    from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

    # Tokenizing the text: finding the unique words from all the captions
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(img_desc)
    vocab_size = len(tokenizer.word_index) + 1

    max_length = 35

    # Load the pre-trained model
    model_file = os.path.join(script_directory, 'best_model.h5')
    model = load_model(model_file)
    
    def mapping_to_word(integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def predict_description(model, image, tokenizer, max_length):
        in_text = 'beginning'
        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], max_length)
            desc_predict = model.predict([image, sequence], verbose=0)

            desc_predict = np.argmax(desc_predict)
            word = mapping_to_word(desc_predict, tokenizer)
            if word is None:
                break
            in_text += " " + word
            if word == 'ending':
                break

        return in_text

    def extract_image_features(image_path):
        # Load the VGG16 model
        base_model = VGG16(weights='imagenet')
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
        

        # Load and preprocess the image
        img = Image.open(image_path)
        img = img.convert("RGB")
        img = img.resize((224, 224))
        img = img_to_array(img)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        # Get the image features
        features = model.predict(img)
      

        return features

    def generate_text(image_name):
        image_id = image_name.split('.')[0]
        img_path = os.path.join('media', image_name)
        image = Image.open(img_path)
        image_features = extract_image_features(img_path)
        y_pred = predict_description(model, image_features, tokenizer, max_length)

        return str(y_pred)

    text = str(generate_text(image_path))
    return text

def upload_image(request):
    if request.method == 'POST':
        form = ImageCaptionForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.save()
            user_language = form.cleaned_data.get('language')

            # Get the image path and generate the caption
            image_path = os.path.join('', str(image.image))
            caption = generate_caption(image_path)
            # caption = caption.replace('beginning', '').replace('ending', '')
            res = caption.split(' ', 1)[1]
            text = res.rsplit(' ', 1)[0]
            text_S = str(text)

            # def translate_to_language(text, dest_language):
            #     translator = Translator()
            #     translated = translator.translate(text, src='en', dest=dest_language)
            #     return translated.text
            
            from translate import Translator

            def translate_to_language(text, target_language):
                translator = Translator(to_lang=target_language)
                try:
                     translated_text = translator.translate(text)
                     return translated_text  
                    
                except Exception as e:
                    print(f"Error during translation: {e}")
                    return text  # Fallback to original text

         
                                    
            translated_text = translate_to_language(text, user_language)
            print(translated_text)
            
            ###### ONLINE TRANSLATIONS USING GOOGLE#######

            tts = gTTS(translated_text)
            tts.save('app/static/info.wav')
            
            
            
            ##### OFFLINE TRANSLATION USING PYTTSX3#####
            # import pyttsx3
            # engine = pyttsx3.init()
            # engine.save_to_file(translated_text, 'app/static/info.wav')
            # engine.runAndWait()
            
                       
            # sound_file = 'info.wav'
            # Audio(sound_file, autoplay=True)
            # Update the image model with the generated caption
            image.caption = text
            image.Tcaption = translated_text
            image.language = user_language
            image.save()
        
            return render(request, 'upload.html', {'form': form,'image':image})
        else:
            return render(request, 'upload.html', {'form': form, 'error_message': 'Please provide input for all required fields.'})
    else:
        form = ImageCaptionForm()

    return render(request, 'upload.html', {'form': form})
