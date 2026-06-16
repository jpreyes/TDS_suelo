# Pipeline Observado

## Ejecutar

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m tsd_suelo build --records-dir C:\Respaldos\records --flatfiles-dir C:\Respaldos\records\flatfiles --output-dir outputs
```

Para una prueba rapida:

```powershell
python -m tsd_suelo build --records-dir C:\Respaldos\records --flatfiles-dir C:\Respaldos\records\flatfiles --output-dir outputs\smoke_real --max-h5 10
```

## Convenciones De Claves

La clave canonica `event_id` sale del nombre del H5:

```text
YYYYMMDDHHMMSS_STATION.h5
```

El identificador interno `EventID_BM16` del H5 se conserva como `h5_event_id_bm16`, pero no se usa para unir contra flatfiles porque no siempre coincide con `EventID` de los CSV.

## Targets H5

Los H5 se interpretan como aceleracion en `cm/s2` y se exportan tambien en `g`.

Targets principales:

```text
PGA E/N/Z/H
Arias E/N/Z/H
CAV E/N/Z/H
duracion 5-75 y 5-95
energia espectral por bandas
frecuencia dominante
centroide y ancho espectral
PSA 5% en T = 0.1, 0.2, 0.5, 1.0, 2.0 s
polarizacion horizontal
razones H/V y E/N
```

## Salida Para Atlas

`atlas_geologico.geojson` contiene features de:

```text
receptores
fuentes
rutas
campos Kozyrev de mayor delta
```

`atlas_geologico.kmz` contiene la misma informacion en formato KML comprimido para inspeccion rapida.

