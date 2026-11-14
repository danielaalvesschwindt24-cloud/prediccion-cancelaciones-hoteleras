import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd


def ejecutar_predicciones(ruta_entrada, ruta_modelo, carpeta_salida):
    artefacto = joblib.load(ruta_modelo)
    modelo = artefacto["modelo"]
    version = artefacto["version"]
    umbral = artefacto["umbral"]
    columnas = artefacto["variables"]

    reservas = pd.read_csv(ruta_entrada)

    faltantes = set(columnas) - set(reservas.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {sorted(faltantes)}")

    reservas = reservas[columnas]

    salida = Path(carpeta_salida)
    salida.mkdir(parents=True, exist_ok=True)

    resultados = []
    registros = []

    for numero, (_, reserva) in enumerate(reservas.iterrows(), start=1):
        inicio = time.perf_counter()

        entrada = reserva.to_frame().T.infer_objects()
        probabilidad = float(modelo.predict_proba(entrada)[0, 1])
        prediccion = int(probabilidad >= umbral)

        tiempo_ms = (time.perf_counter() - inicio) * 1000

        resultados.append({
            "id_caso": numero,
            "probabilidad_cancelacion": probabilidad,
            "prediccion_cancelacion": prediccion,
            "version_modelo": version
        })

        registros.append({
            "fecha_hora": datetime.now(timezone.utc).isoformat(),
            "version_modelo": version,
            "id_caso": numero,
            "resumen_entrada": json.dumps(
                reserva.to_dict(),
                ensure_ascii=False,
                default=str
            ),
            "prediccion": prediccion,
            "tiempo_ejecucion_ms": round(tiempo_ms, 3)
        })

    pd.DataFrame(resultados).to_csv(
        salida / "predicciones.csv", index=False
    )

    pd.DataFrame(registros).to_csv(
        salida / "bitacora.csv", index=False
    )

    print(f"Reservas procesadas: {len(reservas)}")
    print(f"Predicciones guardadas en: {salida / 'predicciones.csv'}")
    print(f"Bitácora guardada en: {salida / 'bitacora.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predicción de cancelaciones hoteleras por lotes"
    )
    parser.add_argument("--entrada", required=True)
    parser.add_argument(
        "--modelo",
        default="modelo/hotel_cancelaciones_v1.joblib"
    )
    parser.add_argument("--salida", default="resultados")

    args = parser.parse_args()

    ejecutar_predicciones(args.entrada, args.modelo, args.salida)
