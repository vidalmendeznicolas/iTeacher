raspi = True
ctrBlank = False
ctrDetect = False
confianza = 0.86
imagen_size = 640
import os
import cv2
import numpy as np
if raspi:
    from picamera2 import Picamera2
import time
from ultralytics import YOLO
import mediapipe as mp
if not raspi:
    import pyttsx3
else:
    from playsound import playsound

current_directory = os.path.dirname(__file__)
pathLinux = model_path = os.path.join(current_directory, 'Model', 'TFGv8.pt') #"/home/nicov/Desktop/TFG/TFGV8.pt"
pathSonLinux = model_path = os.path.join(current_directory, 'Sound', 'mano_detectada.mp3')#"/home/nicov/Desktop/TFG/mano_detectada.mp3"
pathWindows = model_path = os.path.join(current_directory, 'Model', 'TFGv8.pt')#"C:\\Users\\nicov\\Desktop\\UDC\\TFG\\TFGv8.pt"


cv2.namedWindow("Detect", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Detect", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# Variables que controlan o procesamento de YOLO
last_yolo_time = 0
yolo_interval = 1  # Procesar solo un frame por segundo

# Función para convertir o texto a voz en tempo real
def texto_a_voz_en_tiempo_real(texto):
    if not raspi:
        engine.say(texto)
        engine.runAndWait()

# Inicializar o motor de pyttsx3
if not raspi:
    engine = pyttsx3.init()
    # Configurar propiedades da fala
    rate = engine.getProperty('rate')  # Obter la velocidad da fala actual
    engine.setProperty('rate', rate - 50)  # Axustar a velocidad da fala

    volume = engine.getProperty('volume')  # Obter o volumen actual
    engine.setProperty('volume', volume + 0.25)  # Axustar o volumen (entre 0.0 e 1.0)

    voice = engine.getProperty('voices')  # Obter as voces dispoñibles
    engine.setProperty('voice', voice[0].id)  # Establecer a voz (0 para voz masculina, 1 para voz femenina)

mpManos = mp.solutions.hands
manos = mpManos.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.9,
                      min_tracking_confidence=0.8)
mpDibujar = mp.solutions.drawing_utils

start_time = None
hand_detected = False

if raspi:
    model = YOLO(pathLinux)
    # Configuración da cámara
    picam2 = Picamera2()
    config = picam2.preview_configuration.main
    config.size = (imagen_size, imagen_size)
    config.format = "RGB888"
    config.align()
    picam2.configure("preview")
    picam2.start()
else:
    model = YOLO(pathWindows)
    capture = cv2.VideoCapture(0)

while True:

    start_time = time.time()  # Inicia o temporizador
    if raspi:
        frame = picam2.capture_array()
    else:
        ret, frame = capture.read()

    frame = cv2.resize(frame, (imagen_size, imagen_size))
    if ctrBlank:
        result_image = np.full(frame.shape, 255, dtype=np.uint8)  # Creamos unha imaxe con fondo branco
    else:
        result_image = frame

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = manos.process(frameRGB)

    height, width, _ = frame.shape

    # Debuxar o recadro na parte superior esquerda
    region_x_max = int(width * 0.3)  #  30% do ancho desde a esquerda
    region_y_max = int(height * 0.4)  # 40% da altura desde a parte superior

    # Calcular cantos rectángulos caben en cada dirección
    num_x = width // region_x_max
    num_y = height // region_y_max

    # Debuxar os rectángulos en todas as posicións
    for i in range(num_x + 1):  # +1 para cubrir o caso donde non chega completamente a imaxe
        for j in range(num_y + 1):
            # Calcular as coordenadas x e y do recadro
            x_start = i * region_x_max
            y_start = j * region_y_max
            x_end = min(x_start + region_x_max, width)  # Asegurarse de non pasar os límites da imaxe
            y_end = min(y_start + region_y_max, height)

            # Debuxar o rectángulo en result_image
            cv2.rectangle(result_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    if resultado.multi_hand_landmarks:
        for handLms in resultado.multi_hand_landmarks:
            # Obter as coordenadas do punto medio da man
            cx = int(handLms.landmark[mpManos.HandLandmark.WRIST].x * width)
            cy = int(handLms.landmark[mpManos.HandLandmark.WRIST].y * height)

            # Comprobar se a man está na rexión superior esquerda
            if cx < region_x_max and cy < region_y_max:
                if not hand_detected:
                    start_timeHand = time.time()
                    hand_detected = True
                elif time.time() - start_timeHand >= 3:
                    cv2.putText(result_image, "MANO DETECTADA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                cv2.LINE_AA)
                    texto = "Mano Detectada"
                    ctrDetect = not ctrDetect
                    if not raspi:
                        texto_a_voz_en_tiempo_real(texto)
                    else:
                        playsound(pathSonLinux)

                mpDibujar.draw_landmarks(result_image, handLms, mpManos.HAND_CONNECTIONS)
            else:
                hand_detected = False
                start_timeHand = None
    else:
        hand_detected = False
        start_timeHand = None

    # Debuxar os márxenes da imaxe orixinal na imaxe en branco
    height, width, _ = frame.shape
    margin_color = (0, 0, 255)  # Color vermello para os márxenes
    thickness = 2

    cv2.line(result_image, (0, 0), (width, 0), margin_color, thickness)  # Línea superior
    cv2.line(result_image, (0, 0), (0, height), margin_color, thickness)  # Línea esquerda
    cv2.line(result_image, (0, height - 1), (width, height - 1), margin_color, thickness)  # Línea inferior
    cv2.line(result_image, (width - 1, 0), (width - 1, height), margin_color, thickness)  # Línea dereita

    current_time = time.time()
    if ctrDetect:
        # Solo procesar con YOLO se pasou un segundo desde a última vez, se usamos a Raspi
        if raspi:
            if current_time - last_yolo_time >= yolo_interval:
                resultados = model.predict(frame, imgsz=imagen_size, conf=confianza)
                last_yolo_time = current_time  # Actualiza o tiempo da última detección
        else:
            resultados = model.predict(frame, imgsz=imagen_size, conf=confianza)

        # Debuxar as anotacións na imaxe en branco
        for r in resultados:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f'{r.names[cls]} {conf:.2f}'

                # Debuxar o rectángulo do box
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(result_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Calcular o punto medio do box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Debuxar o punto medio
                cv2.circle(result_image, (cx, cy), 5, (255, 0, 0), -1)  # Marcado como un círculo azul

    cv2.imshow("Detect", result_image)
    # Espera pola tecla 'q' para sair do bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    if cv2.waitKey(1) & 0xFF == ord('b'):
        ctrBlank = not ctrBlank
    if cv2.waitKey(1) & 0xFF == ord('d'):
        ctrDetect = not ctrDetect
        if ctrDetect:
            cv2.putText(result_image, "DETECTANDO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
