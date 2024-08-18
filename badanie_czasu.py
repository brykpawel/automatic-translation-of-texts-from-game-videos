import cv2 
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import easyocr
import time
from transformers import pipeline, T5ForConditionalGeneration, AutoTokenizer
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re

generator = pipeline("translation", model="mt5-base-translator-en-pl")


def jaccard_similarity(groundtrouth, recognition):

    groundtrouth = set(groundtrouth.lower().split())
    recognition = set(recognition.lower().split())
    
    intersection_size = len(groundtrouth.intersection(recognition))
    union_size = len(groundtrouth.union(recognition))

    if union_size != 0 :
        similarity = intersection_size / union_size
    
    else: 
        similarity = 0

    return similarity

def wer(ground_trouth, predicted):
    # Split the reference and hypothesis into words
    ref_words = ground_trouth.split()
    hyp_words = predicted.split()
    
    substitutions = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp)
    deletions = len(ref_words) - len(hyp_words)
    insertions = len(hyp_words) - len(ref_words)
	# Total number of words in the reference text
    total_words = len(ref_words)
	# Calculating the Word Error Rate (WER)
    wer = (substitutions + deletions + insertions) / total_words
    return wer


def process_video_with_bbox(video_path, video_type="dialog", speed = 1):
    essa = 1
    control = 0
    cap = cv2.VideoCapture(video_path)
    # pobierz liczbę fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps", fps)
    selected = 0
    select_again = "tak"
    prefix = "grammar: "
    model = T5ForConditionalGeneration.from_pretrained("T5-large-spell")
    tokenizer = AutoTokenizer.from_pretrained("T5-large-spell")
    text_prev = ""
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    font_path = "arial.ttf"
    font_size = 20

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    frame_count = 0
    frame_nr = 40
    i = 0
    reader = easyocr.Reader(['en'])

    translate_rows = []
    text_rows = []
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        if selected == 1 and select_again == "tak":
            select_again = input("Czy chcesz jeszcze raz zaznaczyć ROI? (Odpowiedz 'nie', 'tak') ").strip().lower()
            if select_again == 'tak':
                frame_nr = int(input("Podaj numer ramki: "))
    
            frame_nr = int(frame_nr)
            print("Wybrano numer ramki: ", frame_nr)
            cv2.destroyAllWindows()
            if select_again == 'nie':
                selected = 1
            else:
                selected = 0
            frame_count = 0
            cap = cv2.VideoCapture(video_path)

        frame_count += 1

        # zaznaczanie ROI
        if frame_count > frame_nr  and selected == 0:
            # # Wyświetlanie klatki z bbox i tekstem
            r = cv2.selectROI("select the area", frame)
            cropped_image = frame[int(r[1]):int(r[1]+r[3]),  
                      int(r[0]):int(r[0]+r[2])]
            frame_count = 0
            selected = 1
            cv2.imshow("Frame with BBox and Text", cropped_image)
            if cv2.waitKey(speed) & 0xFF == ord('q'):
                break
        elif selected == 1 and select_again == 'nie':
            break
        elif selected == 0 and select_again != 'tak':
            print("wystąpił błąd")
        else:
            pass
            
    # przetwarzanie frame po poprawnym zaznaczeniu ROI
    if selected == 1 and select_again == 'nie':
        for k in range(0, 6):
            if k == 0:
                print("EasyOCR")
                cap = cv2.VideoCapture(video_path)
                i = 0
                start_time = time.time()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cropped_image = frame[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
                    results = reader.readtext(cropped_image)
                    i += 1
                    if i > 6000:
                        break


                print("time frame EasyOCR: ", (time.time() - start_time)/i)
                print("FPS EasyOCR: ", 1/((time.time() - start_time)/i))
                    
            if k == 6:
                print("EasyOCR z imshow")
                cap = cv2.VideoCapture(video_path)
                i = 0
                start_time = time.time()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cropped_image = frame[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
                    results = reader.readtext(cropped_image)
                    i += 1
                    # wyswietl ramkę na najkrótszy czas
                    cv2.imshow("frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                print("time frame EasyOCR z wyswietlaniem: ", (time.time() - start_time)/i)
                print("FPS EasyOCR z wyswietlaniem: ", 1/((time.time() - start_time)/i))
            if k == 1:
                print("preprocessing")
                cap = cv2.VideoCapture(video_path)
                i = 0
                start_time = time.time()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cropped_image = frame[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
                    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    normalized_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
                    _, thresholded_image = cv2.threshold(normalized_image, 0, 230, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    results = reader.readtext(cropped_image)
                    i += 1
                    if i > 6000:
                        break
                    # wyswietl ramkę na najkrótszy czas
                    # cv2.imshow("frame", frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break

                print("time frame EasyOCR with preprocessing: ", (time.time() - start_time)/i)
                print("FPS EasyOCR with preprocessing: ", 1/((time.time() - start_time)/i))

            if k == 2:
                print("sortowanie")
                cap = cv2.VideoCapture(video_path)
                i = 0
                start_time = time.time()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cropped_image = frame[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
                    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    normalized_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
                    _, thresholded_image = cv2.threshold(normalized_image, 0, 230, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    results = reader.readtext(cropped_image)
                    if len(results) > 1:
                        # print("len:", len(results))
                        df = pd.DataFrame(results, columns=['bbox', 'text', 'conf'])
                        if not df.empty:
                            df['bbox_y1'] = df['bbox'].apply(lambda x: x[0][1])
                            df_sorted = df.sort_values(by='bbox_y1')
                            if not df_sorted.empty:
                                df_list = []
                                current_df = []
                                
                                current_min_y1 = df_sorted.loc[0, 'bbox_y1']

                                for idx, row in df_sorted.iterrows():
                                    if row['bbox_y1'] - current_min_y1 > 10:
                                        df_list.append(pd.DataFrame(current_df))
                                        current_df = [row]
                                        current_min_y1 = row['bbox_y1']
                                    else:
                                        current_df.append(row)
                                        current_min_y1 = df_sorted.loc[idx, 'bbox_y1']

                                if current_df:
                                    df_list.append(pd.DataFrame(current_df))

                                df_list2 = []
                                for j, df_part in enumerate(df_list):
                                    df_part['bbox_x1'] = df_part['bbox'].apply(lambda x: x[0][0])
                                    df_sorted = df_part.sort_values(by='bbox_x1')
                                    df_list2.append(df_sorted)

                                word_sequences = [' '.join(df_part['text'].tolist()) for df_part in df_list2]

                                text = ""
                                for k, sequence in enumerate(word_sequences):
                                    # text_rows.append(sequence)
                                    text = text + " " + sequence
                                # jac = jaccard_similarity(text_prev, text)
                                bleu = sentence_bleu([text], text_prev, smoothing_function=SmoothingFunction().method1)
                                # were = wer(text, text_prev)
                                if bleu < 0.2 or abs(len(text_prev) - len(text)) > 5:
                                    different = True
                                    text_prev = text
                                    control = 1
                                    word_sequences_prev = word_sequences
                                else:
                                    different = False

                                if video_type == "dialog":
                                    if different:
                                        text_dialog = []
                                        for k, sequence in enumerate(word_sequences):
                                            # print("Tekst wykryty", sequence)
                                            sequence = sequence.replace('$', ' s')
                                            sequence = sequence.replace('|', 'I')
                                            sequence = sequence.replace('0', 'O')
                                            sequence = re.sub(r'[^a-zA-Z\s:]', '', sequence)
 
                    i += 1
                    if i > 6000:
                        break
                print("time frame EasyOCR with postprocessing: ", (time.time() - start_time)/i)
                print("FPS EasyOCR with postprocessing ", 1/((time.time() - start_time)/i))
            if k == 3:
                print("poprawiająca NLP")
                cap = cv2.VideoCapture(video_path)
                i = 0
                start_time = time.time()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cropped_image = frame[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
                    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    normalized_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
                    _, thresholded_image = cv2.threshold(normalized_image, 0, 230, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    results = reader.readtext(cropped_image)
                    if len(results) > 1:
                        # print("len:", len(results))
                        df = pd.DataFrame(results, columns=['bbox', 'text', 'conf'])
                        if not df.empty:
                            df['bbox_y1'] = df['bbox'].apply(lambda x: x[0][1])
                            df_sorted = df.sort_values(by='bbox_y1')
                            if not df_sorted.empty:
                                df_list = []
                                current_df = []
                                
                                current_min_y1 = df_sorted.loc[0, 'bbox_y1']

                                for idx, row in df_sorted.iterrows():
                                    if row['bbox_y1'] - current_min_y1 > 10:
                                        df_list.append(pd.DataFrame(current_df))
                                        current_df = [row]
                                        current_min_y1 = row['bbox_y1']
                                    else:
                                        current_df.append(row)
                                        current_min_y1 = df_sorted.loc[idx, 'bbox_y1']

                                if current_df:
                                    df_list.append(pd.DataFrame(current_df))

                                df_list2 = []
                                for j, df_part in enumerate(df_list):
                                    df_part['bbox_x1'] = df_part['bbox'].apply(lambda x: x[0][0])
                                    df_sorted = df_part.sort_values(by='bbox_x1')
                                    df_list2.append(df_sorted)

                                word_sequences = [' '.join(df_part['text'].tolist()) for df_part in df_list2]

                                text = ""
                                for k, sequence in enumerate(word_sequences):
                                    # text_rows.append(sequence)
                                    text = text + " " + sequence
                                # jac = jaccard_similarity(text_prev, text)
                                bleu = sentence_bleu([text], text_prev, smoothing_function=SmoothingFunction().method1)
                                # were = wer(text, text_prev)
                                if bleu < 0.2 or abs(len(text_prev) - len(text)) > 5:
                                    different = True
                                    text_prev = text
                                    control = 1
                                    word_sequences_prev = word_sequences
                                else:
                                    different = False

                                if video_type == "dialog":
                                    if different:
                                        text_dialog = []
                                        for k, sequence in enumerate(word_sequences):
                                            # print("Tekst wykryty", sequence)
                                            sequence = sequence.replace('$', ' s')
                                            sequence = sequence.replace('|', 'I')
                                            sequence = sequence.replace('0', 'O')
                                            sequence = re.sub(r'[^a-zA-Z\s:]', '', sequence)
                                            if len(sequence) > 2:
                                                sentence = prefix + sequence
                                                encodings = tokenizer(sentence, return_tensors="pt")
                                                generated_tokens = model.generate(**encodings, max_length=100, early_stopping=True)
                                                answer = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
 
                    i += 1
                    if i > 6000:
                        break
                print("time frame EasyOCR with poprawiająca NLP: ", (time.time() - start_time)/i)
                print("FPS EasyOCR with poprawiająca NLP", 1/((time.time() - start_time)/i))
            if k == 4:
                print("poprawiająca NLP z tłumaczeniem")
                cap = cv2.VideoCapture(video_path)
                i = 0
                start_time = time.time()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cropped_image = frame[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
                    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    normalized_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
                    _, thresholded_image = cv2.threshold(normalized_image, 0, 230, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    results = reader.readtext(cropped_image)
                    if len(results) > 1:
                        # print("len:", len(results))
                        df = pd.DataFrame(results, columns=['bbox', 'text', 'conf'])
                        if not df.empty:
                            df['bbox_y1'] = df['bbox'].apply(lambda x: x[0][1])
                            df_sorted = df.sort_values(by='bbox_y1')
                            if not df_sorted.empty:
                                df_list = []
                                current_df = []
                                
                                current_min_y1 = df_sorted.loc[0, 'bbox_y1']

                                for idx, row in df_sorted.iterrows():
                                    if row['bbox_y1'] - current_min_y1 > 10:
                                        df_list.append(pd.DataFrame(current_df))
                                        current_df = [row]
                                        current_min_y1 = row['bbox_y1']
                                    else:
                                        current_df.append(row)
                                        current_min_y1 = df_sorted.loc[idx, 'bbox_y1']

                                if current_df:
                                    df_list.append(pd.DataFrame(current_df))

                                df_list2 = []
                                for j, df_part in enumerate(df_list):
                                    df_part['bbox_x1'] = df_part['bbox'].apply(lambda x: x[0][0])
                                    df_sorted = df_part.sort_values(by='bbox_x1')
                                    df_list2.append(df_sorted)

                                word_sequences = [' '.join(df_part['text'].tolist()) for df_part in df_list2]

                                text = ""
                                for k, sequence in enumerate(word_sequences):
                                    # text_rows.append(sequence)
                                    text = text + " " + sequence
                                # jac = jaccard_similarity(text_prev, text)
                                bleu = sentence_bleu([text], text_prev, smoothing_function=SmoothingFunction().method1)
                                # were = wer(text, text_prev)
                                if bleu < 0.2 or abs(len(text_prev) - len(text)) > 5:
                                    different = True
                                    text_prev = text
                                    control = 1
                                    word_sequences_prev = word_sequences
                                else:
                                    different = False

                                if video_type == "dialog":
                                    if different:
                                        text_dialog = []
                                        for k, sequence in enumerate(word_sequences):
                                            # print("Tekst wykryty", sequence)
                                            sequence = sequence.replace('$', ' s')
                                            sequence = sequence.replace('|', 'I')
                                            sequence = sequence.replace('0', 'O')
                                            sequence = re.sub(r'[^a-zA-Z\s:]', '', sequence)
                                            if len(sequence) > 2:
                                                sentence = prefix + sequence
                                                encodings = tokenizer(sentence, return_tensors="pt")
                                                generated_tokens = model.generate(**encodings, max_length=100, early_stopping=True)
                                                answer = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                                                translating = generator(answer[0], max_length=512)
 
                    i += 1
                    if i > 6000:
                        break
                print("time frame EasyOCR with poprawiająca NLP z tłumaczeniem: ", (time.time() - start_time)/i)
                print("FPS EasyOCR with poprawiająca NLP z tłumaczeniem", 1/((time.time() - start_time)/i))
            if k == 5:
                print("koncowo z różnicami")
                cap = cv2.VideoCapture(video_path)
                i = 0
                start_time = time.time()
                i = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cropped_image = frame[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
                    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    normalized_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
                    _, thresholded_image = cv2.threshold(normalized_image, 0, 230, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    results = reader.readtext(cropped_image)
                    # print("results", results)
                    if len(results) > 1:
                        # print("len:", len(results))
                        df = pd.DataFrame(results, columns=['bbox', 'text', 'conf'])
                        if not df.empty:
                            df['bbox_y1'] = df['bbox'].apply(lambda x: x[0][1])
                            df_sorted = df.sort_values(by='bbox_y1')
                            if not df_sorted.empty:
                                df_list = []
                                current_df = []
                                
                                current_min_y1 = df_sorted.loc[0, 'bbox_y1']

                                for idx, row in df_sorted.iterrows():
                                    if row['bbox_y1'] - current_min_y1 > 10:
                                        df_list.append(pd.DataFrame(current_df))
                                        current_df = [row]
                                        current_min_y1 = row['bbox_y1']
                                    else:
                                        current_df.append(row)
                                        current_min_y1 = df_sorted.loc[idx, 'bbox_y1']

                                if current_df:
                                    df_list.append(pd.DataFrame(current_df))

                                df_list2 = []
                                for j, df_part in enumerate(df_list):
                                    df_part['bbox_x1'] = df_part['bbox'].apply(lambda x: x[0][0])
                                    df_sorted = df_part.sort_values(by='bbox_x1')
                                    df_list2.append(df_sorted)

                                word_sequences = [' '.join(df_part['text'].tolist()) for df_part in df_list2]

                                text = ""
                                for k, sequence in enumerate(word_sequences):
                                    # text_rows.append(sequence)
                                    text = text + " " + sequence
                                # jac = jaccard_similarity(text_prev, text)
                                bleu = sentence_bleu([text], text_prev, smoothing_function=SmoothingFunction().method1)
                                # were = wer(text, text_prev)
                                if bleu < 0.2 or abs(len(text_prev) - len(text)) > 5:
                                    different = True
                                    text_prev = text
                                    control = 1
                                    word_sequences_prev = word_sequences
                                else:
                                    different = False

                                if video_type == "dialog":
                                    if different:
                                        text_dialog = []
                                        for k, sequence in enumerate(word_sequences):
                                            # print("Tekst wykryty", sequence)
                                            sequence = sequence.replace('$', ' s')
                                            sequence = sequence.replace('|', 'I')
                                            sequence = sequence.replace('0', 'O')
                                            sequence = re.sub(r'[^a-zA-Z\s:]', '', sequence)
                                            # sequence = re.sub(r'[^a-zA-Z0-9\s:]', '', sequence)
                                            if len(sequence) > 2:
                                                sentence = prefix + sequence
                                                encodings = tokenizer(sentence, return_tensors="pt")
                                                generated_tokens = model.generate(**encodings, max_length=100, early_stopping=True)
                                                answer = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                                                translating = generator(answer, max_length=512)
                                        
                                        # print("tłumaczono", "tłumaczenie:", text_dialog, "time:", time.time() - start_time)            
                    i += 1
                    if i > 6000:
                        break
                print("time frame koncowo z roznicami: ", (time.time() - start_time)/i)
                print("FPS koncowo z roznicami:  ", 1/((time.time() - start_time)/i))
   

 

print("COD 2")
video_path = "COD_eng_2.mov"
process_video_with_bbox(video_path, "dialog",1)

print("wiedzmin")
video_path = "wiedzmin_eng.mov"
process_video_with_bbox(video_path, "dialog",1)

print("dr_house_english_T")
video_path = "dr_house_english_T.mov"
process_video_with_bbox(video_path, "dialog",1)

print("the_office_english_T")
video_path = "the_office_english_T.mov"
process_video_with_bbox(video_path, "dialog",1)

print("good_doctor_english_T")
video_path = "good_doctor_english_T.mov"
process_video_with_bbox(video_path, "dialog",1)



