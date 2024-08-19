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


def process_video_with_bbox(video_path, video_type="dialog", speed = 1, csv_translation = "tlumaczenie.csv", csv_text = "text_eng.csv"):
    
    control = 0
    cap = cv2.VideoCapture(video_path)
    selected = 0
    select_again = "tak"
    prefix = "grammar: "
    model = T5ForConditionalGeneration.from_pretrained("T5-large-spell")
    tokenizer = AutoTokenizer.from_pretrained("T5-large-spell")
    text_prev = ""
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    frame_nr = 40
    i = 0
    reader = easyocr.Reader(['en'])
    whole_df = pd.DataFrame()
    whole_df2 = pd.DataFrame()
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
            
        # przetwarzanie frame po poprawnym zaznaczeniu ROI
        elif selected == 1 and select_again == 'nie':
            cropped_image = frame[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
            if i:
            # cv2.imshow("frame", cropped_image)
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
                            bleu = sentence_bleu([text], text_prev, smoothing_function=SmoothingFunction().method1)
                            
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
                                            text_dialog.append(translating[0])
                                            # text_rows.append(sequence)
                                            text_rows.append(answer[0])
                                            translate_rows.append(translating[0])
                                    
                                    print("tłumaczono", "tłumaczenie:", text_dialog, "time:", time.time() - start_time)
                                else:
                                    # print("nie tłumaczono")
                                    pass
                                    
                                #wyświetl ramkę z naniesionym tłumaczeniem
                                # rgb = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
                                # draw = ImageDraw.Draw(rgb)
                                # draw.text((10, 10), "Test", font=font, fill=(255, 255, 255, 255))
                                # gray = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

                                # cv2.putText(gray, text['translation_text'].values[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                cv2.imshow("frame", frame)
                                
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                                
                            else:
                                if different:
                                    # print(text, "ramka:", i)
                                    text = text.replace('$', ' s')
                                    text = text.replace('|', 'I')
                                    text = text.replace('0', 'O')
                                    text = re.sub(r'[^a-zA-Z\s:]', '', text)
                                    # text = re.sub(r'[^a-zA-Z0-9\s:]', '', text)
                                    if len(text) > 2:
                                        # text_rows.append(text)
                                        sentence = prefix + text
                                        encodings = tokenizer(sentence, return_tensors="pt")
                                        generated_tokens = model.generate(**encodings, max_length=500, early_stopping=True)
                                        answer = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                                        text_rows.append(answer[0])
                                        # print(answer[0], bleu)
                                        translating = generator(answer[0], max_length=1024)
                                        
                                        translate_rows.append(translating[0])
                                        print("Tłumaczono ", "zawartość tekstowa: ", text, "tłumaczenie: ", translating[0]['translation_text'], "bleu: ", bleu, "różnica długości: ", abs(len(text_prev) - len(text)))
                                else:
                                    # print("Nie tłumaczono")
                                    pass
                                                                    
                                # wyświetl ramkę z naniesionym tłumaczeniem
                                # cv2.putText(gray, translating[0]['translation_text'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                cv2.imshow("frame", frame)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                                           
            i += 1
            roi_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(roi_pil)
            draw.text((10, 10), "Test", font=font, fill=(255, 255, 255, 255))

            # Konwersja z powrotem na obraz OpenCV
            roi = cv2.cvtColor(np.array(roi_pil), cv2.COLOR_RGB2BGR)
            cv2.imshow("Frame ROI", roi)
            if cv2.waitKey(speed) & 0xFF == ord('q'):
                break

        elif selected == 0 and select_again != 'tak':
            print("wystąpił błąd")
        else:
            pass
        #     if cv2.waitKey(speed) & 0xFF == ord('q'):
        #         break

        frame_count += 1
    if frame is not None:
        width = frame.shape[1]
        height = frame.shape[0]
    else:
        width = 0
        height = 0
    frame_count = 0
    cap.release()
    cv2.destroyAllWindows()
    whole_df = pd.concat([whole_df, pd.DataFrame(translate_rows)], ignore_index=True)
    whole_df2 = pd.concat([whole_df2, pd.DataFrame(text_rows)], ignore_index=True)

    
    whole_df.to_csv(csv_text, index=False, header=False, sep=';', encoding='utf-8-sig')
    whole_df2.to_csv(csv_translation, index=False, header=False, sep=';', encoding='utf-8-sig')
    return width, height

def main():
    video_path = "dr_house_english_T.mov"
    process_video_with_bbox(video_path, "dialog", 1, "tlumaczenie.csv", "text_eng.csv")

main()