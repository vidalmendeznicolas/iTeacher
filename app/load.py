import tkinter as tk
from tkinter import PhotoImage, Button
import subprocess
import os
import sys

current_directory = os.path.dirname(__file__)

def run_script(script_path):
    # Cerrar la ventana actual
    root.destroy()
    # Ejecutar el script de Python con la ruta completa
    subprocess.Popen([sys.executable, script_path])

# Crear la ventana principal
root = tk.Tk()
root.title("SAR Application Control")

# Establecer el tamaño de la ventana
#root.geometry("800x600")

# Configurar la ventana para iniciar en pantalla completa
#root.attributes('-fullscreen', True)
#root.wm_attributes('-fullscreen', 'true')  # Intenta forzar el modo de pantalla completa.
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root.overrideredirect(True)  # Esto elimina la barra de título y bordes.

# Cargar y mostrar la imagen de fondo
pathImage = os.path.join(current_directory,'Images', 'Fig8.3.png')
background_image = PhotoImage(file=pathImage)
background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Crear botones
button1 = Button(root, text="Run Main", command=lambda: run_script(os.path.join(current_directory, 'main.py')),
                 height=2, width=20, bg='light blue')  # Color azul claro)
button1.place(x=150, y=450)  # Posicionar el botón 'Run Main' en (x=50, y=100)
#button1.pack(side='left', padx=10, pady=20, fill='both', expand=False)

button2 = Button(root, text="Run Calibration", command=lambda: run_script(os.path.join(current_directory, 'calibration.py')),
                 height=2, width=20, bg='light green')  # Color verde claro)
button2.place(x=150, y=550)
#button2.pack(side='left', padx=10, pady=20, fill='both', expand=False)

button3 = Button(root, text="Exit", command=root.destroy,
                 height=2, width=20, bg='salmon')  # Color salmón
button3.place(x=150, y=650)
#button3.pack(side='left', padx=10, pady=20, fill='both', expand=False)

# Iniciar el bucle principal de Tkinter
root.mainloop()
