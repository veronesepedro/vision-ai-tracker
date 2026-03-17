import cv2
import os
from core.processor import GestureProcessor
from core.constants import MP_MODEL_PATH, CUSTOM_MODEL_PATH, ENCODER_PATH

def main():
    # Verifica se os modelos existem
    if not all(os.path.exists(p) for p in [MP_MODEL_PATH, CUSTOM_MODEL_PATH, ENCODER_PATH]):
        print("Erro: Um ou mais arquivos de modelo não foram encontrados:")
        print(f"- {MP_MODEL_PATH}: {'OK' if os.path.exists(MP_MODEL_PATH) else 'MISSING'}")
        print(f"- {CUSTOM_MODEL_PATH}: {'OK' if os.path.exists(CUSTOM_MODEL_PATH) else 'MISSING'}")
        print(f"- {ENCODER_PATH}: {'OK' if os.path.exists(ENCODER_PATH) else 'MISSING'}")
        return

    print("--- Inicializando Processador de Gestos ---")
    processor = GestureProcessor(MP_MODEL_PATH, CUSTOM_MODEL_PATH, ENCODER_PATH)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nIniciando reconhecimento CUSTOMIZADO... Pressione 'q' para sair.")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Função solicitada: recebe uma imagem e retorna uma imagem
            processed_frame = processor.process_frame(frame)

            cv2.imshow('Custom Gesture Recognition', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        processor.close()

if __name__ == "__main__":
    main()


