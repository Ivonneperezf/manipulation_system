import cv2
import numpy as np
import pyrealsense2 as rs

path = "/home/user/Documentos/ROS/projects/kinova_robot_ws/src/kinova-ros/calibration/config"
# Parámetros del patrón
pattern_size = (8, 6)  # número de esquinas internas: columnas x filas
square_size = 25.0     # tamaño de cada cuadro en mm

# Coordenadas 3D reales del patrón (z=0)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# Listas de puntos para la calibración
objpoints = []
imgpoints = []

# Configurar la cámara RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipeline.start(config)
print("Cámara RealSense D415 iniciada. Presiona 's' para guardar imagen válida. 'q' para salir.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detectar esquinas del patrón
        found, corners = cv2.findChessboardCorners(gray, pattern_size)

        display = color_image.copy()
        if found:
            cv2.drawChessboardCorners(display, pattern_size, corners, found)
            cv2.putText(display, "Patron detectado - Presiona 's' para guardar", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No se detecta el patron", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Calibracion RealSense D415", display)
        key = cv2.waitKey(10)

        if key == ord('s') and found:
            objpoints.append(objp.copy())
            imgpoints.append(corners)
            print(f"Imagen capturada. Total capturas: {len(objpoints)}")

        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# Guardar los datos en archivo
if len(objpoints) > 0:
    np.savez(f"{path}/camera_calibration.npz", objpoints=objpoints, imgpoints=imgpoints, image_shape=gray.shape[::-1])
    print(f"Se guardaron {len(objpoints)} capturas en 'camera_calibration.npz'.")
else:
    print("No se capturaron imágenes válidas.")