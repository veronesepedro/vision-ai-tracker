import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import collections

class GestureProcessor:
    def __init__(self, mp_model_path, custom_model_path, encoder_path):
        # Configurações do MediaPipe
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=mp_model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.recognizer = GestureRecognizer.create_from_options(options)
        
        # Carrega o modelo customizado e o encoder de labels
        self.clf = joblib.load(custom_model_path)
        self.label_encoder = joblib.load(encoder_path)
        
        # Utilitários de desenho
        self.mp_hands = mp.tasks.vision.HandLandmarksConnections
        self.mp_drawing = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles

        # Histórico para estabilização (últimas 7 predições por mão)
        self.history = collections.defaultdict(lambda: collections.deque(maxlen=7))

    def process_frame(self, frame, draw_landmarks=True):
        """
        Recebe uma imagem (BGR) e retorna a mesma imagem anotada se draw_landmarks for True.
        """
        # Prepara a imagem para o MediaPipe
        frame_rgb = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Timestamp necessário para o modo VIDEO
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        
        # Extrai landmarks usando MediaPipe
        recognition_result = self.recognizer.recognize_for_video(mp_image, timestamp_ms)

        labels = []

        if recognition_result.hand_landmarks:
            for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
                # 1. Desenha os landmarks (se solicitado)
                if draw_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame_rgb,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())

                # 2. Prepara dados para o modelo customizado
                hand_label = recognition_result.handedness[i][0].category_name
                handedness_val = 0 if hand_label == 'Left' else 1
                
                landmarks_array = [handedness_val]
                for lm in hand_landmarks:
                    landmarks_array.extend([lm.x, lm.y, lm.z])
                
                features = np.array(landmarks_array).reshape(1, -1)
                
                # Predição do modelo customizado
                prediction_idx = self.clf.predict(features)[0]
                prediction_prob = np.max(self.clf.predict_proba(features))
                raw_gesture_name = self.label_encoder.inverse_transform([prediction_idx])[0]

                # Filtro de moda (majority vote) para evitar oscilações
                self.history[hand_label].append(raw_gesture_name)
                gesture_name = collections.Counter(self.history[hand_label]).most_common(1)[0][0]

                # Adiciona o label do modelo na resposta em vez de desenhar
                labels.append({
                    "hand": hand_label,
                    "gesture": gesture_name,
                    "confidence": float(prediction_prob)
                })
        
        return frame_rgb, labels

    def close(self):
        self.recognizer.close()
