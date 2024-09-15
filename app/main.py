ctrBlank = False
raspi = True
confianza = 0.82
imagen_size = 640
x_max = 0.3
y_max = 0.4
estado = 0
mostrar_flecha = True
import cv2
import numpy as np
import os
if raspi:
    from picamera2 import Picamera2
import time
from ultralytics import YOLO
import mediapipe as mp
if not raspi:
    import pyttsx3
else:
    from playsound import playsound
#import tkinter as tk
#from tkinter import PhotoImage, Label
current_directory = os.path.dirname(__file__)
pathLinux = os.path.join(current_directory, 'Model', 'TFGv8.pt') #"/home/nicov/Desktop/TFG/TFGV8.pt"
pathSonLinux = os.path.join(current_directory, 'Sound', 'mano_detectada.mp3')#"/home/nicov/Desktop/TFG/mano_detectada.mp3"
pathWindows = os.path.join(current_directory, 'Model', 'TFGv8.pt')#"C:\\Users\\nicov\\Desktop\\UDC\\TFG\\TFGv8.pt"


cv2.namedWindow("Detect", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Detect", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Variables para controlar o procesamento de YOLO
last_yolo_time = 0
yolo_interval = 1  # Procesar solo un frame por segundo

# Función para crear y ejecutar el thread
#def iniciar_thread(estado, raspi):
#    thread = threading.Thread(target=ejecutar_voz, args=(estado, raspi))
#    thread.start()
#    thread.join()  # Esperar a que el thread termine para que se elimine automáticamente

def ejecutar_voz(estado, raspi):
    if estado == 0:
        texto = "Bienvenido a iTeacher, para comenzar con el tutorial ponga su mano encima de la zona remarcada durante tres segundos"
        if not raspi:
            texto_a_voz_en_tiempo_real(texto)
        else:
            playsound(os.path.join(current_directory, 'Sound', 'estado0.mp3')) #playsound("/home/nicov/Desktop/TFG/estado0.mp3")
    if estado == 1:
        texto = "Coloque el pegamento en las zonas remarcadas, una vez terminado, vuelva a posicionar su mano en la zona remarcada para continuar"
        if not raspi:
            texto_a_voz_en_tiempo_real(texto)
        else:
            playsound(os.path.join(current_directory, 'Sound', 'estado1.mp3'))#playsound("/home/nicov/Desktop/TFG/estado1.mp3")
    if estado == 2:
        texto = "Por último, en las zonas resaltadas, perfore con la broca del 5, una vez terminado posicione su mano en la zona remarcada para volver a empezar"
        if not raspi:
            texto_a_voz_en_tiempo_real(texto)
        else:
            playsound(os.path.join(current_directory, 'Sound', 'estado2.mp3'))#playsound("/home/nicov/Desktop/TFG/estado2.mp3")

# Función para convertir texto a voz en tempo real
def texto_a_voz_en_tiempo_real(texto):
    if not raspi:
        engine.say(texto)
        engine.runAndWait()

def cambiar_estado(parEstado):
    global estado
    global raspi
    match parEstado:
        case 0:
            estado = 1
        case 1:
            estado = 2
        case 2:
            estado = 1
        case _:
            estado = 0
    ejecutar_voz(estado, raspi)
    cv2.putText(result_image, "Estado {}".format(estado) , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                cv2.LINE_AA)
    time.sleep(1.5)

# Inicializar o motor de pyttsx3
if not raspi:
    engine = pyttsx3.init()
    # Configurar propiedades da fala
    rate = engine.getProperty('rate')  # Obter a velocidad da fala actual
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

if estado == 0:
    # Cargar a imaxe de fondo e mostrala con OpenCV
    background_image_path = os.path.join(current_directory, 'Images', 'principal.jpg')
    img = cv2.imread(background_image_path)
    cv2.namedWindow('Background', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Background', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Background', img)
    cv2.waitKey(1)  # Actualizar a ventana
    ejecutar_voz(estado, raspi)
    # Pechar a ventana coa imaxe de fondo
    cv2.destroyAllWindows()
    cv2.namedWindow("Detect", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Detect", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    tiempo_anterior = time.time()

while True:
    # Espera pola tecla 'q' para sair do bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    if cv2.waitKey(1) & 0xFF == ord('b'):
        ctrBlank = not ctrBlank

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
    region_x_max = int(width * x_max)
    region_y_max = int(height * y_max)
    cv2.rectangle(result_image, (0, 0), (region_x_max, region_y_max), (0, 255, 0), 2)
    if estado == 0:
        start_timeFecla = time.time()

        # Coordenadas da flecha apuntando hacia a esquerda
        start_point = (region_x_max + 50, region_y_max // 2)  # 50 píxeles a dereita do recadro
        end_point = (region_x_max + 20, region_y_max // 2)  # 10 píxeles a dereita do recadro

        # Definir puntos de inicio e fin para a flecha debaixo do recadro apuntando cara arriba
        start_point_bottom = (region_x_max // 2, region_y_max + 50)  # Debaixo do recadro
        end_point_bottom = (region_x_max // 2, region_y_max + 20)  # Justo debaixo do recadro
        # Verificar se pasou un segundo desde a última vez que se alternou a visibilidade
        if start_timeFecla - tiempo_anterior >= 1:
            mostrar_flecha = not mostrar_flecha  # Alternar a visibilidad da flecha
            tiempo_anterior = start_timeFecla  # Actualizar o tempo de referencia

        # Debuxar ou borrar a flecha según o estado de `mostrar_flecha`
        if mostrar_flecha:
            cv2.arrowedLine(result_image, start_point, end_point, (0, 0, 255), 5, tipLength=0.5)
            cv2.arrowedLine(result_image, start_point_bottom, end_point_bottom, (0, 0, 255), 5, tipLength=0.5)
        else:
            # Borrar a flecha debuxando sobre ela co color de fondo
            cv2.arrowedLine(result_image, start_point, end_point, (255, 255, 255), 5, tipLength=0.5)
            cv2.arrowedLine(result_image, start_point_bottom, end_point_bottom, (255, 255, 255), 5, tipLength=0.5)

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

                    # Crear unha imaxe en branco
                    result_image = np.full(frame.shape, 255, dtype=np.uint8)

                    # Convertir a imaxe a escala de grises
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

                    # Convertir de novo a BGR para poder poñer texto en color
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)

                    # Definir o texto, fonte, escala e cor
                    text = "Audio en curso"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_color = (0, 255, 0)  # Cor verde en BGR
                    thickness = 2

                    # Obter o tamaño do texto para centralo
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

                    # Calcular as coordenadas do texto para centralo
                    text_x = (result_image.shape[1] - text_size[0]) // 2
                    text_y = (result_image.shape[0] + text_size[1]) // 2

                    # Poñer o texto na imaxe
                    cv2.putText(result_image, text, (text_x, text_y), font, font_scale, font_color, thickness)
                    cv2.imshow("Detect", result_image)
                    cambiar_estado(estado)


                mpDibujar.draw_landmarks(result_image, handLms, mpManos.HAND_CONNECTIONS)
            else:
                hand_detected = False
                start_timeHand = None
    else:
        hand_detected = False
        start_timeHand = None

    # Debuxar os márxenes da imaxe orixinal na imaxe en branco
    height, width, _ = frame.shape
    margin_color = (0, 0, 255)  # Cor vermello para os márxenes
    thickness = 2

    cv2.line(result_image, (0, 0), (width, 0), margin_color, thickness)  # Línea superior
    cv2.line(result_image, (0, 0), (0, height), margin_color, thickness)  # Línea esquerda
    cv2.line(result_image, (0, height - 1), (width, height - 1), margin_color, thickness)  # Línea inferior
    cv2.line(result_image, (width - 1, 0), (width - 1, height), margin_color, thickness)  # Línea dereita

    current_time = time.time()

    if estado >= 1:
        # Solo procesar con YOLO se pasou un segundo desde a última vez, se usamos a Raspi
        if raspi:
            if current_time - last_yolo_time >= yolo_interval:
                resultados = model.predict(frame, imgsz=imagen_size, conf=confianza)
                last_yolo_time = current_time  # Actualiza o tiempo da última detección
        else:
            resultados = model.predict(frame, imgsz=imagen_size, conf=confianza)

        # Debuxar as anotaciones na imaxe en branco
        for r in resultados:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f'{r.names[cls]} {conf:.2f}'
                #cv2.putText(result_image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            #cv2.LINE_AA)
                print("label:", label)
                if estado == 1 and "CASA" in label:
                    # Debuxar o rectángulo do box
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(result_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Calcular o punto medio do box
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Debuxar o punto medio
                    cv2.circle(result_image, (cx, cy), 80, (255, 0, 0), -1)  # Marcado como un círculo azul
                if estado == 2 and "PONTE" in label:
                    # Debuxar o rectángulo do box
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(result_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Calcular o punto medio do box
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Debuxar o punto medio
                    cv2.circle(result_image, (cx, cy), 50, (0, 0, 255), -1)  # Marcado como un círculo rojo
                    cv2.circle(result_image, (cx, cy), 5, (255, 0, 0), -1)  # Marcado como un círculo azul

    cv2.imshow("Detect", result_image)

