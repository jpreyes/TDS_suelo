# Plan De Implementacion TSD-Suelo

## Principio De Independencia

El repositorio solo consume:

```text
records/*.h5
records/flatfiles/*.csv
```

No se importan scripts, parquets, caches, coeficientes, runners ni paquetes internos de GMPE, Modelo E o TSD estructural.

## Capas

| Fase | Objetivo | Producto |
|---|---|---|
| F00 | Inventario observado | `observed_inventory.json` |
| F01 | Geometria fuente-receptor | `record_geometry.parquet` |
| F02 | Indice receptor/suelo | `receiver_index.parquet` |
| F03 | Indice fuente 3D | `source3d_index.parquet` |
| F04 | Rutas multiescala | `route_graph_observed.parquet` |
| F05 | Targets H5 observados | `waveform_targets_observed.parquet` |
| F06 | Tabla geologica observada | `geo_targets_observed.parquet` |
| F07 | Residualizacion | `geo_residuals.parquet`, `target_level_attribution.csv` |
| F08 | Modos latentes | `latent_modes.parquet`, `latent_mode_components.csv` |
| F09 | Kozyrev/grafo | `kozyrev_graph_fields.parquet` |
| F10 | Atlas | `atlas_geologico.geojson`, `atlas_geologico.kmz` |

## Residualizacion

La base removida es:

```text
evento/fuente + Mw + profundidad + distancia + sitio conocido
```

El sitio conocido incluye:

```text
Vs30
f0_HVSR
A0_HVSR
kappa0
pendiente topografica
```

No se usa efecto fijo de estacion en esta etapa porque podria borrar la senal de suelo que se busca descubrir.

## Kozyrev Multiescala

Cada observacion genera celdas:

```text
source_cell_j1..j4
receiver_cell_j1..j4
route_id_j1..j4
```

Los campos Kozyrev se calculan como deltas coarse-to-fine de los modos latentes:

```text
delta_Jk = modo_promedio_Jk - modo_promedio_padre_J(k-1)
```

Esto deja candidatos a discontinuidades, cuencas, rutas anomalas y anisotropia direccional sin inventar geologia externa.

