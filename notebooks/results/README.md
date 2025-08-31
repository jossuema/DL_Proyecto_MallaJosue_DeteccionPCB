
# Resultados del Proyecto de Detección PCB

## Resumen de Archivos

- `metricas_comparativas.csv`: Tabla comparativa de métricas principales
- `resultados_detallados.json`: Métricas completas y metadatos
- `inferencias/`: Capturas de inferencias en imágenes de test
- `graficas/`: Visualizaciones de comparación de modelos

## Modelos Evaluados

1. **YOLOv8s**
1. **YOLOv8m**
1. **YOLOv8l**

## Dataset

- **Fuente**: Roboflow Universe - Printed Circuit Board
- **Clases**: 31 (Battery, Button, Buzzer, Capacitor, Capacitor Jumper, Clock, Connector, Diode, Display, EM, Electrolytic Capacitor, Ferrite Bead, Fuse, Heatsink, IC, Inductor, Jumper, Led, PS, Pads, Pins, Potentiometer, Resistor, Resistor Jumper, Resistor Network, SK, Switch, Test Point, Transformer, Transistor, Zener Diode)
- **Total de imágenes**: 1069
  - Entrenamiento: 794
  - Validación: 200
  - Prueba: 75