import numpy as np
import cv2
import os

# =========================
# Ruta donde guardaste datos
# =========================
path = "/home/user/Documentos/ROS/projects/kinova_robot_ws/src/kinova-ros/calibration/config"
data_file = os.path.join(path, "camera_calibration.npz")

# =========================
# Cargar datos capturados
# =========================
data = np.load(data_file, allow_pickle=True)

objpoints = [np.array(pts, dtype=np.float32) for pts in data["objpoints"]]
imgpoints = [np.array(pts, dtype=np.float32) for pts in data["imgpoints"]]
image_shape = tuple(data["image_shape"])  # (width, height)

print("Número de imágenes válidas:", len(objpoints))

# =========================
# Ejecutar calibración
# =========================
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    image_shape,
    None,
    None
)

# =========================
# Mostrar resultados
# =========================
print("\n=== MATRIZ INTRÍNSECA ===")
print(camera_matrix)

print("\n=== COEFICIENTES DE DISTORSIÓN ===")
print(dist_coeffs.ravel())

# =========================
# Calcular error de reproyección
# =========================
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )

    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

mean_error /= len(objpoints)

print("\n=== ERROR DE REPROYECCIÓN PROMEDIO ===")
print(mean_error)

# =========================
# Guardar en formato para Hand-Eye
# =========================
output_file = os.path.join(path, "camera_final_calibration.npz")

np.savez(
    output_file,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs
)

print(f"\nCalibración guardada en:\n{output_file}")