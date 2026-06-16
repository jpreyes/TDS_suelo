# TSD-Suelo

TSD-Suelo es un sistema independiente para identificar dinamica compatible del suelo desde observaciones. No es un GMPE clasico y no hereda scripts, parquets, runners ni dependencias internas de GMPE, Modelo E o TSD estructural.

Fuentes primarias permitidas:

```text
C:\Respaldos\records\*.h5
C:\Respaldos\records\flatfiles\*.csv
```

## Objetivo

Descubrir fallas, rupturas, cuencas, rutas anomalas, anisotropia direccional y comportamiento de suelo mediante una lectura multiescala fuente-ruta-receptor:

```text
fuente 3D -> ruta -> receptor/suelo -> targets fisicos observados -> residuos -> modos latentes -> grafo Kozyrev -> atlas
```

El forward condicionado queda preparado como contrato posterior, pero no se ajusta ni genera acelerogramas sinteticos en esta etapa.

## Instalacion local

```powershell
cd C:\Respaldos\TDS_suelo
python -m pip install -e .[dev]
```

Tambien se puede ejecutar sin instalar:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m tsd_suelo build --records-dir C:\Respaldos\records --flatfiles-dir C:\Respaldos\records\flatfiles --output-dir outputs
```

## CLI

```powershell
tsd-suelo inventory
tsd-suelo targets --max-h5 20
tsd-suelo build --records-dir C:\Respaldos\records --flatfiles-dir C:\Respaldos\records\flatfiles --output-dir outputs
```

## Productos

El build completo escribe productos derivados en `outputs/`:

```text
observed_inventory.json
waveform_targets_observed.parquet
record_geometry.parquet
receiver_index.parquet
source3d_index.parquet
geo_targets_observed.parquet
geo_residuals.parquet
target_level_attribution.csv
latent_modes.parquet
latent_mode_components.csv
route_graph_observed.parquet
kozyrev_graph_fields.parquet
atlas_geologico.geojson
atlas_geologico.kmz
forward_conditioning_template.json
pipeline_manifest.json
```

`outputs/` esta ignorado por git porque son artefactos reproducibles.

## Flujo implementado

1. Inventario observado de H5, eventos, records y estaciones.
2. ETL desde H5 y flatfiles.
3. Geometria fuente-receptor: distancia, azimut, backazimut, incidencia y celdas multiescala.
4. Targets fisicos desde H5: PGA, Arias, CAV, duraciones, energia por bandas, frecuencia dominante, centroides y PSA aproximada 5%.
5. Tabla `geo_targets_observed.parquet`.
6. Residualizacion por fuente/evento, magnitud, profundidad, distancia y sitio conocido (`Vs30`, HVSR, `kappa0`, pendiente).
7. Modos latentes por PCA sobre residuos fisicos.
8. Proyeccion en grafo Kozyrev fuente 3D -> ruta -> receptor.
9. Atlas geologico GeoJSON/KMZ.
10. Plantilla de forward condicionado posterior.

## Tests

```powershell
pytest
```

Las pruebas usan H5 y flatfiles sinteticos temporales; no dependen de los datos reales.

