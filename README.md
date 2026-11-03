<p align="center">
  <img width="400" height="400" alt="Chispas" src="https://github.com/user-attachments/assets/21f5c975-cfe5-4fd9-867c-1c27c7ab149f" />
</p>

<h1 align="center">Chispas ✦</h1>

<p align="center">
  Componentes TypeScript para <strong>8th Wall</strong>, con la experiencia de configuración que extrañabas de Meta Spark.
</p>

<p align="center">
  <a href="#instalación">Instalación</a> ·
  <a href="#chispas-disponibles">Chispas disponibles</a> ·
  <a href="#de-meta-spark-a-8th-wall">Migrar desde Spark</a> ·
  <a href="#contribuir">Contribuir</a>
</p>

---

## ¿Qué es Chispas?

Cuando Meta Spark cerró, muchos creadores nos quedamos sin herramientas que daban por resuelto lo difícil: segmentar a una persona, ocultar objetos detrás de ella, seguir manos o rostros… todo con un par de variables en un panel.

**Chispas** recrea esa experiencia para [8th Wall open source](https://8thwall.org). Cada *chispa* es un componente TypeScript que agregas a tu proyecto y configuras con propiedades simples, sin escribir shaders ni pelearte con el pipeline de la cámara.

- 🔌 **Plug and play:** copias el componente, lo registras y lo configuras desde propiedades.
- 🎛️ **Variables seleccionables:** opciones al estilo Spark en vez de código a medida.
- 📱 **Pensado para gama media:** fluido en celulares comunes de Latinoamérica, sin pedirle al usuario que active flags de Chrome ni elija calidad.
- 🆓 **Libre:** úsalo en proyectos personales, educativos o comerciales.

## Chispas disponibles

| Chispa | Qué hace | Estado |
| --- | --- | --- |
| **Segmentación** | Separa a la persona del fondo usando la cámara. | ✅ Disponible |
| **Oclusión** | Hace que la persona tape los objetos virtuales que están detrás de ella. | 🛠️ En desarrollo |
| **Hand tracking** | Sigue la mano para anclar o controlar contenido. | Investigando |
| **Sky Segmentation** | Seguimiento de rostro para máscaras y efectos. | 🗓️ Planeado |

> ¿Te falta alguna? Abre un [issue](../../issues) con la función de Spark que más extrañas.

## Requisitos

- Un proyecto de [8th Wall open source](https://8thwall.org) funcionando.
- TypeScript.
- Un celular con cámara para probar (navegador móvil actualizado).

## Instalación

1. Descarga o clona este repositorio:

   ```bash
   git clone https://github.com/emiliusvgs/chispas.git
   ```

2. Copia la carpeta de la chispa que necesitas (por ejemplo `src/segmentacion/`) dentro de tu proyecto de 8th Wall.

3. Regístrala y agrégala a tu escena siguiendo el README de esa chispa, donde están sus propiedades y un ejemplo mínimo.

> 💡 Cada chispa es independiente: puedes usar solo la que necesites.

## Estructura del repositorio

```
chispas/
├── src/
│   ├── segmentacion/     # Chispa de segmentación de persona
│   ├── oclusion/         # Chispa de oclusión
│   └── ...               # Próximas chispas
├── examples/             # Proyectos de ejemplo listos para probar
├── assets/               # Logo y recursos gráficos
└── README.md
```

## De Meta Spark a 8th Wall

Si vienes de Spark, esta tabla te ayuda a ubicarte:

| En Meta Spark usabas… | En Chispas usas… |
| --- | --- |
| Person Segmentation | Chispa de **Segmentación** |
| Segmentación + oclusión de persona | Chispa de **Oclusión** |


📺 Tutorial en video: https://youtu.be/GXxYblIDbLU

## Rendimiento

Todas las chispas se prueban en celulares de gama media, no solo en equipos de gama alta. La meta es que la experiencia se vea bien y fluida sin que el usuario tenga que tocar nada. Si en tu equipo va lento, abre un issue indicando modelo de celular y navegador.

## Contribuir

¡Las contribuciones son bienvenidas!

1. Haz un fork del repositorio.
2. Crea una rama: `git checkout -b mi-chispa`.
3. Haz tus cambios y pruébalos en un celular real.
4. Abre un pull request explicando qué agrega o corrige.

También ayuda muchísimo reportar bugs, proponer chispas nuevas o compartir lo que construyes con ellas.

## Historia

Chispas nació en 2026 para ayudar a quienes migraban sus efectos de Meta Spark a 8th Wall open source. Empezó con la segmentación, siguió con la oclusión, y la idea es que crezca hacia una colección de componentes de realidad aumentada para la web, más allá de su origen.

## Autor

Creado por **Emilio Vegas** ([EMILIUSVGS](https://emiliusvgs.com/) · Laboratorio del Futuro).

## Licencia

Distribuido bajo licencia MIT. Consulta el archivo [LICENSE](./LICENSE).

---

<sub>Chispas es un proyecto independiente y no está afiliado, patrocinado ni avalado por Meta Platforms, Inc. ni por Niantic Spatial, Inc. "Meta Spark" y "8th Wall" son marcas de sus respectivos dueños y se mencionan solo con fines descriptivos.</sub>
