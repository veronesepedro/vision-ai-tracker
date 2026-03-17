import os
import json
from fasthtml.common import *
from starlette.websockets import WebSocket, WebSocketDisconnect
from starlette.staticfiles import StaticFiles

from core.processor import GestureProcessor
from core.constants import MP_MODEL_PATH, CUSTOM_MODEL_PATH, ENCODER_PATH
from core.utils import decode_image, encode_image
import time


# Inicialização mínima do FastHTML sem estilos padrão
app, rt = fast_app(pico=False)

app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Instanciando o processador de imagens uma única vez
processor = None
if all(os.path.exists(p) for p in [MP_MODEL_PATH, CUSTOM_MODEL_PATH, ENCODER_PATH]):
    processor = GestureProcessor(MP_MODEL_PATH, CUSTOM_MODEL_PATH, ENCODER_PATH)

@rt("/")
def get():
    return (
        Title("AI Gesture Recognition"),
        Link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap"),
        Link(rel="stylesheet", href="/assets/style.css"),
        Main(
            Div(
                # Grid principal com os dois painéis
                Div(
                    # Painel da Esquerda: Imagem ao Vivo
                    Div(
                        Div(
                            Span("IMAGEM AO VIVO", cls="panel-title"),
                            cls="panel-header"
                        ),
                        Div(
                            Div("Connected", cls="badge badge-connected"),
                            Div("0 fps", id="fps-counter", cls="badge badge-fps"),
                            cls="status-bar"
                        ),
                        Div(
                            Video(id="video", autoplay=True, style="display:none;"),
                            Canvas(id="canvas"),
                            cls="webcam-container"
                        ),
                        cls="panel"
                    ),
                    # Painel da Direita: Gesto Detectado
                    Div(
                        Div(
                            Span("GESTO DETECTADO", cls="panel-title"),
                            cls="panel-header"
                        ),
                        Div(
                            Div(
                                Span("Nenhum Gesto", id="gesture-name", cls="gesture-name"),
                                cls="gesture-name-container"
                            ),
                            Span("0% confiança", id="confidence-text", cls="confidence-text"),
                            Div(
                                Img(src="/assets/images/gestures/paz.png", id="gesture-match-image", style="display:none;"),
                                cls="illustration-box"
                            ),
                            cls="gesture-result"
                        ),
                        cls="panel"
                    ),
                    cls="main-grid"
                ),
                # Barra de Controle inferior
                Div(
                    Div(
                        Span("Qualidade da imagem", cls="control-label"),
                        Span("100%", id="quality-value", cls="control-value"),
                        Input(type="range", id="quality-slider", min="0.01", max="1.0", step="0.01", value="1.0"),
                        cls="control-item"
                    ),
                    Div(
                        Span("Framerate", cls="control-label"),
                        Span("30 fps", id="fps-limit-value", cls="control-value"),
                        Input(type="range", id="fps-limit-slider", min="1", max="60", step="1", value="30"),
                        cls="control-item"
                    ),
                    Div(
                        Span("Mostrar anotações na imagem ao vivo", cls="control-label"),
                        Label(
                            Input(type="checkbox", id="landmarks-toggle", checked=True),
                            Span(cls="slider"),
                            cls="switch"
                        ),
                        cls="control-item"
                    ),
                    cls="control-bar"
                ),
                cls="app-container"
            ),
            cls="view-container"
        ),
        Script(src="/assets/script.js")
    )

@app.websocket_route('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_time = time.time()
    try:
        while True:
            data = await websocket.receive_text()
            
            # Tenta decodificar como JSON para pegar configurações
            try:
                payload = json.loads(data)
                img_data = payload.get("image", "")
                draw_landmarks = payload.get("draw_landmarks", True)
            except json.JSONDecodeError:
                img_data = data
                draw_landmarks = True

            if img_data.startswith("data:image"):
                img = decode_image(img_data)

                labels = []
                if processor:
                    img, labels = processor.process_frame(img, draw_landmarks=draw_landmarks)

                image_to_show = None
                if len(labels) == 2 and labels[0]["gesture"] == labels[1]["gesture"]:
                    gesture = labels[0]["gesture"].lower()
                    image_to_show = f"{gesture}.png"

                out_dataURL = encode_image(img)
                
                current_time = time.time()
                fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
                last_time = current_time

                response_data = {
                    "image": out_dataURL,
                    "labels": labels,
                    "image_to_show": image_to_show,
                    "fps": round(fps, 1)
                }

                await websocket.send_text(json.dumps(response_data))
    except WebSocketDisconnect:
        pass

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)
