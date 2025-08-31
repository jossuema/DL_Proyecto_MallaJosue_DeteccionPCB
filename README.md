# Proyecto de Detección de Objetos en PCB: YOLOv8 vs RT-DETR

Este proyecto académico compara el rendimiento de diferentes modelos de detección de objetos (YOLOv8) en la detección de componentes en placas de circuito impreso (PCB).

## 📁 Estructura del Proyecto

```
DL_Proyecto_TuNombre_Deteccion/
├── notebooks/
│   └── Proyecto_Deteccion.ipynb    # Notebook principal de Google Colab
├── results/                         # Carpeta para resultados (inicialmente vacía)
│   ├── inferencias/                # Capturas de inferencias
│   ├── graficas/                   # Gráficas comparativas
│   ├── metricas_comparativas.csv   # Tabla de métricas
│   └── resultados_detallados.json  # Métricas completas
├── README.md                       # Este archivo
├── requirements.txt                # Dependencias del proyecto
└── Informe_Proyecto.docx          # Informe académico editable
```

## 🎯 Objetivo

Comparar el rendimiento de modelos de detección de objetos en la tarea de detección de defectos en PCB, evaluando métricas como mAP@0.5, mAP@0.5:0.95, precision, recall y velocidad de inferencia.

## 📊 Dataset

- **Fuente**: [Roboflow Universe - Printed Circuit Board](https://universe.roboflow.com/roboflow-100/printed-circuit-board)
- **Licencia**: CC BY 4.0 (uso académico permitido)
- **Clases**: 31 tipos de componentes electronicos

| Clase (EN)                | Descripción (ES)                |
|--------------------------|----------------------------------|
| **Battery**              | Batería                          |
| **Button**               | Botón                            |
| **Buzzer**               | Zumbador                         |
| **Capacitor**            | Condensador                      |
| **Capacitor Jumper**     | Condensador puente               |
| **Clock**                | Reloj                            |
| **Connector**            | Conector                         |
| **Diode**                | Diodo                            |
| **Display**              | Pantalla                         |
| **EM**                   | Componente EM                    |
| **Electrolytic Capacitor**| Condensador electrolítico        |
| **Ferrite Bead**         | Núcleo de ferrita                |
| **Fuse**                 | Fusible                          |
| **Heatsink**             | Disipador                        |
| **IC**                   | Circuito integrado               |
| **Inductor**             | Inductor                         |
| **Jumper**               | Puente                           |
| **Led**                  | LED                              |
| **PS**                   | Fuente de poder (Power Supply)   |
| **Pads**                 | Pads                             |
| **Pins**                 | Pines                            |
| **Potentiometer**        | Potenciómetro                    |
| **Resistor**             | Resistor                         |
| **Resistor Jumper**      | Resistor puente                  |
| **Resistor Network**     | Red de resistores                |
| **SK**                   | Componente SK                    |
| **Switch**               | Interruptor                      |
| **Test Point**           | Punto de prueba                  |
| **Transformer**          | Transformador                    |
| **Transistor**           | Transistor                       |
| **Zener Diode**          | Diodo Zener                      |


## 🤖 Modelos Evaluados

1. **YOLOv8s**
2. **YOLOv8m**
3. **YOLOv8L**

## 🚀 Cómo Ejecutar

### 1. Preparación del Entorno

1. Abre Google Colab
2. Sube el notebook `notebooks/Proyecto_Deteccion.ipynb`
3. Asegúrate de tener GPU habilitada (Runtime → Change runtime type → GPU)

### 2. Ejecución

1. **Setup**: Ejecuta las celdas de instalación de librerías
2. **Dataset**: Configura tu API key de Roboflow y descarga el dataset
3. **Entrenamiento**: Ejecuta el entrenamiento de ambos modelos (puede tomar 2-4 horas)
4. **Evaluación**: Ejecuta la evaluación completa con todas las métricas
5. **Inferencia**: Realiza inferencia en imágenes de test
6. **Resultados**: Los resultados se guardarán automáticamente en `/content/results/`

### 3. Configuración de Roboflow API

```python
# En la celda correspondiente, reemplaza YOUR_API_KEY con tu clave real
rf = Roboflow(api_key="TU_API_KEY_AQUI")
```

**Obtener API Key:**
1. Crea cuenta en [Roboflow](https://roboflow.com/)
2. Ve a Account Settings
3. Copia tu API key

## 📈 Métricas de Evaluación

### Métricas Principales
- **mAP@0.5**: Mean Average Precision con IoU threshold de 0.5
- **mAP@0.5:0.95**: mAP promedio para IoU thresholds de 0.5 a 0.95
- **Precision**: Precisión global del modelo
- **Recall**: Recall global del modelo
- **F1-Score**: Media harmónica de precision y recall
- **FPS**: Frames per second (velocidad de inferencia)

### Métricas por Clase
- Precision por cada clase de defecto
- Recall por cada clase de defecto
- AP@0.5 por cada clase de defecto

## 📋 Entrenamiento y Evaluación

### Configuración YOLOv8s
- **Épocas**: 200
- **Batch size**: 16
- **Imagen size**: 1024x1024
- **Optimizer**: AdamW
- **Learning rate**: 0.001
- **Augmentaciones**: Habilitadas (mosaic, mixup, copy_paste)

### Configuración YOLOv8m
- **Épocas**: 200
- **Batch size**: 16
- **Imagen size**: 1024x1024
- **Optimizer**: AdamW
- **Learning rate**: 0.001
- **Augmentaciones**: Habilitadas (mosaic, mixup, copy_paste)

### Configuración YOLOv8l
- **Épocas**: 200
- **Batch size**: 16
- **Imagen size**: 1024x1024
- **Optimizer**: AdamW
- **Learning rate**: 0.001
- **Augmentaciones**: Habilitadas (mosaic, mixup, copy_paste)

## 📊 Tabla Comparativa de Métricas

El notebook generará automáticamente una tabla como esta:

| Modelo | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1-Score | FPS | Tiempo (ms) |
|--------|---------|--------------|-----------|--------|----------|-----|-------------|
| YOLOv8s | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| YOLOv8m | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| YOLOv8l | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

*Los valores se completarán automáticamente tras la ejecución*

## 🔍 Inferencia y Visualización

El proyecto incluye:
- Inferencia en 4 imágenes de test
- Visualización de bounding boxes con clases y scores
- Cálculo de IoU con ground truth
- Comparación visual lado a lado de ambos modelos

## 📁 Resultados Generados

Después de la ejecución, encontrarás:

1. **metricas_comparativas.csv**: Tabla con métricas principales
2. **resultados_detallados.json**: Métricas completas y metadatos
3. **inferencias/**: Capturas de inferencias con anotaciones
4. **graficas/**: Gráficas comparativas y gráfico radar
5. **README.md**: Resumen de resultados


## 📚 Referencias

1. **YOLOv8**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
3. **Dataset**: [Roboflow PCB Dataset](https://universe.roboflow.com/roboflow-100/printed-circuit-board)
4. **Métricas**: [COCO Evaluation Metrics](https://cocodataset.org/#detection-eval)
