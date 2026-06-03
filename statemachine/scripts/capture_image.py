#!/usr/bin/env python3
"""
capture_image.py
----------------
Nodo ROS Noetic que se suscribe al tópico /camera/image_raw.
Mantiene siempre el último frame en memoria y lo muestra en pantalla.
Al presionar Enter en la terminal, pide un nombre y guarda la imagen.
Se puede capturar múltiples veces; escribe 'salir' para cerrar el nodo.
"""

import os
import sys
import threading
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class ImageCapture:
    def __init__(self, save_dir: str):
        self.save_dir = os.path.expanduser(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        self.bridge   = CvBridge()
        self.last_frame = None          # último frame recibido
        self.lock       = threading.Lock()

        topic = rospy.get_param("~topic", "/camera/color/image_raw")
        rospy.loginfo(f"Suscrito a:            {topic}")
        rospy.loginfo(f"Directorio de guardado: {self.save_dir}")
        rospy.loginfo("Presiona Enter para capturar | escribe 'salir' para terminar\n")

        self.sub = rospy.Subscriber(topic, Image, self._cb_image, queue_size=1)

        # Hilo dedicado a la interacción con el usuario (terminal)
        self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._input_thread.start()

    # ------------------------------------------------------------------
    def _cb_image(self, msg: Image):
        """Actualiza el último frame recibido (sin bloquearse)."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.last_frame = frame
        except Exception as e:
            rospy.logerr(f"Error al convertir imagen: {e}")

    # ------------------------------------------------------------------
    def _input_loop(self):
        """Espera Enter, pide nombre y guarda la imagen actual."""
        while not rospy.is_shutdown():
            try:
                # Bloquea hasta que el usuario presione Enter
                entrada = input()
            except EOFError:
                break

            if entrada.strip().lower() == "salir":
                rospy.loginfo("Cerrando nodo...")
                rospy.signal_shutdown("Usuario solicitó salir")
                break

            # Tomamos el frame actual de forma segura
            with self.lock:
                frame = self.last_frame.copy() if self.last_frame is not None else None

            if frame is None:
                rospy.logwarn("⚠  Aún no se ha recibido ningún frame del tópico.")
                continue

            # Pedir nombre al usuario
            try:
                nombre = input("  Nombre de la imagen (sin extensión): ").strip()
            except EOFError:
                break

            if not nombre:
                rospy.logwarn("⚠  Nombre vacío, captura cancelada.")
                continue

            # Sanitizar nombre
            nombre = "".join(c if c.isalnum() or c in "-_." else "_" for c in nombre)

            filepath = os.path.join(self.save_dir, f"{nombre}.png")

            # Evitar sobreescribir un archivo existente
            if os.path.exists(filepath):
                rospy.logwarn(f"⚠  Ya existe '{filepath}'. Elige otro nombre.")
                continue

            if cv2.imwrite(filepath, frame):
                rospy.loginfo(f"✔  Imagen guardada: {filepath}")
            else:
                rospy.logerr(f"✘  No se pudo guardar en: {filepath}")

            print("\nPresiona Enter para capturar | escribe 'salir' para terminar")

    # ------------------------------------------------------------------
    def spin(self):
        """Ciclo principal en el hilo principal para actualizar la ventana de OpenCV."""
        rate = rospy.Rate(30) # 30 FPS para refrescar la ventana
        
        while not rospy.is_shutdown():
            with self.lock:
                frame = self.last_frame.copy() if self.last_frame is not None else None

            if frame is not None:
                # Mostrar la imagen en una ventana de OpenCV
                cv2.imshow("Vista en Vivo (Presiona Enter en terminal)", frame)
            
            # cv2.waitKey(1) es OBLIGATORIO para que la ventana procese los gráficos
            cv2.waitKey(1)
            rate.sleep()
        
        # Al salir, destruir las ventanas de OpenCV de forma limpia
        cv2.destroyAllWindows()


# ======================================================================
def main():
    rospy.init_node("capture_image_node", anonymous=False)

    if len(sys.argv) > 1 and not sys.argv[1].startswith("__"):
        save_dir = sys.argv[1]
    else:
        save_dir = rospy.get_param("~save_dir", "~/capturas_ros")

    node = ImageCapture(save_dir)
    node.spin()


if __name__ == "__main__":
    main()