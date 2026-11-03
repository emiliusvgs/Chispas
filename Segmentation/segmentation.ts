// segmentation.ts — Componente de segmentación para 8th Wall Studio / Desktop
// Equivalente al "Segmentation" de Meta Spark: Cámara → Persona / Fondo / Pelo / Piel / Ropa.
// Usa MediaPipe Tasks Vision de Google (licencia Apache 2.0).
//
// Rendimiento: se usa el modelo más liviano posible según el objetivo:
//   - Persona / Fondo → selfie_segmenter   (~250 KB, muy rápido)
//   - Pelo            → hair_segmenter     (~780 KB, rápido)
//   - Piel / Ropa     → selfie_multiclass  (~16 MB, LENTO: úsalo solo si lo necesitas)
//
// Precisión y cero desfase: el componente dibuja él mismo la imagen de cámara usando
// EXACTAMENTE el mismo frame que analizó MediaPipe, y refina el borde con los colores reales.
//
// Uso: agrega este componente a una entidad vacía. El objetivo se pinta con un color o
// una imagen; Intensidad 1 = lo reemplaza por completo.
//  - Fondo + imagen + intensidad 1 → cambio de fondo.
//  - Pelo + color + intensidad 0.5 → tinte de pelo.

import * as ecs from '@8thwall/ecs'

const {THREE} = window as any

const VERSION = '11 (forma del modelo + color)'

// IMPORTANTE para producción: descarga estos archivos y cámbialos por rutas de tu proyecto.
const MP_VERSION = '0.10.14'
const MP_BUNDLE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.mjs`
const MP_WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`
const MODELOS_BASE = 'https://storage.googleapis.com/mediapipe-models/image_segmenter'

type Modelo = 'selfie' | 'pelo' | 'multiclase'

const MODELOS: Record<Modelo, string> = {
  selfie: `${MODELOS_BASE}/selfie_segmenter/float16/latest/selfie_segmenter.tflite`,
  pelo: `${MODELOS_BASE}/hair_segmenter/float32/latest/hair_segmenter.tflite`,
  multiclase: `${MODELOS_BASE}/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite`,
}

// Esquema híbrido para que todo vaya fluido:
//  - El modelo rápido (selfie) corre en CADA frame y da el contorno de la persona,
//    perfectamente alineado con la imagen.
//  - Para Pelo / Piel / Ropa, el modelo pesado corre en un hilo aparte (Web Worker)
//    a su propio ritmo, sin frenar la cámara. Su resultado se combina con el contorno
//    actual: el borde contra el fondo siempre está al día.
//
// Canales de salida de cada modelo:
//  selfie:     [0] persona
//  pelo:       [0] fondo, [1] pelo
//  multiclase: [0] fondo, [1] pelo, [2] piel cuerpo, [3] piel cara, [4] ropa, [5] otros
type Objetivo = {clase: Modelo | null, canales: number[], complemento: boolean}
const OBJETIVOS: Record<string, Objetivo> = {
  Persona: {clase: null, canales: [], complemento: false},
  Fondo: {clase: null, canales: [], complemento: true},  // fondo = 1 − persona
  Pelo: {clase: 'pelo', canales: [1], complemento: false},
  Piel: {clase: 'multiclase', canales: [2, 3], complemento: false},
  Ropa: {clase: 'multiclase', canales: [4], complemento: false},
}

// ---------- Worker para los modelos pesados ----------

// Código del worker (se crea desde texto, no necesita archivo aparte)
const CODIGO_WORKER = `
  let seg = null
  let canales = []
  self.onmessage = async (e) => {
    const m = e.data
    try {
      if (m.type === 'init') {
        canales = m.canales
        // El CDN sirve el bundle con un tipo que importScripts rechaza: se pasa por un blob
        const txt = await (await fetch(m.bundle)).text()
        self.exports = {}
        self.module = {exports: self.exports}
        importScripts(URL.createObjectURL(new Blob([txt], {type: 'text/javascript'})))
        const vision = self.module.exports
        const fileset = await vision.FilesetResolver.forVisionTasks(m.wasm)
        seg = await vision.ImageSegmenter.createFromOptions(fileset, {
          baseOptions: {modelAssetPath: m.model, delegate: 'CPU'},
          runningMode: 'IMAGE',
          outputCategoryMask: false,
          outputConfidenceMasks: true,
        })
        self.postMessage({type: 'listo'})
      } else if (m.type === 'frame' && seg) {
        const t0 = performance.now()
        const img = new ImageData(new Uint8ClampedArray(m.data), m.w, m.h)
        const res = seg.segment(img)
        const ms = res.confidenceMasks
        const w = ms[0].width
        const h = ms[0].height
        const arrs = canales.filter((i) => i < ms.length).map((i) => ms[i].getAsFloat32Array())
        const out = new Uint8Array(w * h)
        for (let p = 0; p < w * h; p++) {
          let v = 0
          for (let k = 0; k < arrs.length; k++) v += arrs[k][p]
          out[p] = v <= 0 ? 0 : v >= 1 ? 255 : (v * 255) | 0
        }
        if (res.close) res.close()
        self.postMessage({type: 'mascara', w, h, mask: out, ms: performance.now() - t0}, [out.buffer])
      }
    } catch (err) {
      self.postMessage({type: 'error', msg: String((err && err.stack) || err)})
    }
  }
`

// ---------- Máscara de clase (Pelo / Piel / Ropa) ----------
//
// Se combinan dos pistas, cada una cubre el punto débil de la otra:
//  - FORMA: la máscara del modelo de IA (pesado, llega con unos frames de retraso). Se
//    desplaza según cómo se movió la silueta y, en el shader, su borde se reajusta
//    localmente al contorno real de la imagen actual.
//  - COLOR: los colores típicos de la clase aprendidos del propio modelo. Responde al
//    instante, pero SOLO puede ampliar la forma cerca de donde ya estaba (para seguir el
//    movimiento). Colores ambiguos (p. ej. negro en pelo y en pantalón) no deciden nada.
// La silueta actual (modelo rápido) recorta todo: el borde contra el fondo va siempre al día.

// Colores en YCbCr: 8 niveles de brillo × 32×32 de tono
const BY = 8
const BC = 32
const NUM_BINS = BY * BC * BC
const binDe = (r: number, g: number, b: number) => {
  const y = (0.299 * r + 0.587 * g + 0.114 * b) | 0
  const cb = (128 - 0.1687 * r - 0.3313 * g + 0.5 * b) | 0
  const cr = (128 + 0.5 * r - 0.4187 * g - 0.0813 * b) | 0
  return ((y >> 5) << 10) | ((cb >> 3) << 5) | (cr >> 3)
}

// Centro y tamaño de la silueta de la persona
type Silueta = {cx: number, cy: number, tam: number}
const medirSilueta = (persona: Float32Array, w: number, h: number): Silueta | null => {
  let sx = 0
  let sy = 0
  let a = 0
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (persona[y * w + x] > 0.5) {
        sx += x
        sy += y
        a++
      }
    }
  }
  if (a < 50) return null
  return {cx: sx / a, cy: sy / a, tam: Math.sqrt(a)}
}

type Clasificador = {
  worker: Worker
  modelo: Modelo
  listo: boolean
  ocupado: boolean
  // frame, silueta y tamaño enviados (para emparejarlos con la respuesta)
  pendFrame: Uint8ClampedArray | null
  pendPersona: Float32Array | null
  pendPos: Silueta | null
  pendW: number
  pendH: number
  enviadoEn: number
  latencia: number             // ms de ida y vuelta del último análisis
  // colores aprendidos
  hc: Float32Array
  hn: Float32Array
  probabilidad: Float32Array
  version: number
  // última máscara de clase: nítida y ensanchada
  zona: Float32Array | null
  zonaAncha: Float32Array | null
  zonaW: number
  zonaH: number
  zonaPos: Silueta | null
  ms: number
  respuestas: number
}

// Suaviza un histograma 3D con sus vecinos (tolera cambios pequeños de luz)
const suavizarHistograma = (h: Float32Array): Float32Array => {
  const out = new Float32Array(NUM_BINS)
  for (let y = 0; y < BY; y++) {
    for (let cb = 0; cb < BC; cb++) {
      for (let cr = 0; cr < BC; cr++) {
        let s = 0
        for (let dy = -1; dy <= 1; dy++) {
          const yy = y + dy
          if (yy < 0 || yy >= BY) continue
          for (let db = -1; db <= 1; db++) {
            const bb = cb + db
            if (bb < 0 || bb >= BC) continue
            for (let dr = -1; dr <= 1; dr++) {
              const rr = cr + dr
              if (rr < 0 || rr >= BC) continue
              const peso = (dy === 0 ? 2 : 1) * (db === 0 ? 2 : 1) * (dr === 0 ? 2 : 1)
              s += h[(yy << 10) | (bb << 5) | rr] * peso
            }
          }
        }
        out[(y << 10) | (cb << 5) | cr] = s
      }
    }
  }
  return out
}

// Desenfoque de caja separable (radio R) sobre un mapa w×h
const desenfocar = (src: Float32Array, w: number, h: number, R: number): Float32Array => {
  const tmp = new Float32Array(w * h)
  const out = new Float32Array(w * h)
  for (let y = 0; y < h; y++) {
    let s = 0
    let c = 0
    for (let x = 0; x <= Math.min(R, w - 1); x++) { s += src[y * w + x]; c++ }
    for (let x = 0; x < w; x++) {
      tmp[y * w + x] = s / c
      const sale = x - R
      const entra = x + R + 1
      if (sale >= 0) { s -= src[y * w + sale]; c-- }
      if (entra < w) { s += src[y * w + entra]; c++ }
    }
  }
  for (let x = 0; x < w; x++) {
    let s = 0
    let c = 0
    for (let y = 0; y <= Math.min(R, h - 1); y++) { s += tmp[y * w + x]; c++ }
    for (let y = 0; y < h; y++) {
      out[y * w + x] = s / c
      const sale = y - R
      const entra = y + R + 1
      if (sale >= 0) { s -= tmp[sale * w + x]; c-- }
      if (entra < h) { s += tmp[entra * w + x]; c++ }
    }
  }
  return out
}

// Guarda la máscara del modelo pesado y aprende los colores de la clase
const procesarRespuesta = (c: Clasificador, mascara: Uint8Array, w: number, h: number) => {
  const frame = c.pendFrame
  const persona = c.pendPersona
  if (!frame || !persona || c.pendW !== w || c.pendH !== h) return
  const n = w * h

  // Colores: memoria corta, lo nuevo pesa más
  for (let i = 0; i < NUM_BINS; i++) {
    c.hc[i] *= 0.5
    c.hn[i] *= 0.5
  }
  for (let p = 0; p < n; p++) {
    if (persona[p] < 0.5) continue
    const o = p * 4
    const bin = binDe(frame[o], frame[o + 1], frame[o + 2])
    const clase = mascara[p] / 255
    c.hc[bin] += clase
    c.hn[bin] += 1 - clase
  }
  const sc = suavizarHistograma(c.hc)
  const sn = suavizarHistograma(c.hn)
  for (let i = 0; i < NUM_BINS; i++) {
    c.probabilidad[i] = (sc[i] + 0.02) / (sc[i] + sn[i] + 0.2)
  }

  // Forma: nítida + ensanchada (el ancho crece con el retraso del modelo)
  const z = new Float32Array(n)
  for (let p = 0; p < n; p++) z[p] = mascara[p] / 255
  const R = Math.max(4, Math.min(14, Math.round(c.latencia / 40) + 4))
  c.zona = z
  c.zonaAncha = desenfocar(z, w, h, R)
  c.zonaW = w
  c.zonaH = h
  c.zonaPos = c.pendPos
  c.version++
}

const crearClasificador = (modelo: Modelo, canales: number[]): Clasificador => {
  const url = URL.createObjectURL(new Blob([CODIGO_WORKER], {type: 'text/javascript'}))
  const worker = new Worker(url)
  const c: Clasificador = {
    worker,
    modelo,
    listo: false,
    ocupado: false,
    pendFrame: null,
    pendPersona: null,
    pendPos: null,
    pendW: 0,
    pendH: 0,
    enviadoEn: 0,
    latencia: 0,
    hc: new Float32Array(NUM_BINS),
    hn: new Float32Array(NUM_BINS),
    probabilidad: new Float32Array(NUM_BINS),
    version: 0,
    zona: null,
    zonaAncha: null,
    zonaW: 0,
    zonaH: 0,
    zonaPos: null,
    ms: 0,
    respuestas: 0,
  }
  worker.onmessage = (e: MessageEvent) => {
    const m = e.data
    if (m.type === 'listo') {
      c.listo = true
      console.log('[segmentacion] Modelo', modelo, 'listo en hilo aparte')
    } else if (m.type === 'mascara') {
      c.latencia = performance.now() - c.enviadoEn
      procesarRespuesta(c, m.mask, m.w, m.h)
      c.ocupado = false
      c.ms += m.ms
      c.respuestas++
      if (c.respuestas === 30) {
        console.log('[segmentacion] Modelo', modelo, '(hilo aparte):',
          (c.ms / 30).toFixed(0), 'ms por análisis')
        c.ms = 0
        c.respuestas = 0
      }
    } else if (m.type === 'error') {
      c.ocupado = false
      console.error('[segmentacion] Error en el hilo del modelo', modelo, m.msg)
    }
  }
  worker.postMessage({
    type: 'init',
    bundle: MP_BUNDLE.replace('vision_bundle.mjs', 'vision_bundle.cjs'),
    wasm: MP_WASM,
    model: MODELOS[modelo],
    canales,
  })
  return c
}

// Envía el frame reducido al worker si está libre (si está ocupado, se salta)
const enviarAClasificador = (c: Clasificador, img: ImageData, persona: Float32Array) => {
  if (!c.listo || c.ocupado) return
  c.ocupado = true
  c.pendFrame = new Uint8ClampedArray(img.data)
  c.pendPersona = persona.slice()
  c.pendW = img.width
  c.pendH = img.height
  c.pendPos = medirSilueta(persona, img.width, img.height)
  c.enviadoEn = performance.now()
  const copia = img.data.slice().buffer
  c.worker.postMessage({type: 'frame', w: img.width, h: img.height, data: copia}, [copia])
}

// Lado más largo de la imagen que recibe el modelo
const LADO_MODELO = 256

type Delegado = 'GPU' | 'CPU'

// ---------- Frames de cámara desde 8th Wall ----------

const XR_MODULO = 'segmentacion-frames'
const frameCompartido = {
  img: null as ImageData | null,
  id: 0,
  voltear: false,
}
let modulosConectados = false
let resolucionActual = 0

const conectarXR8 = (resolucion: number) => {
  const XR8 = (window as any).XR8
  if (!XR8 || !XR8.addCameraPipelineModule || !XR8.CameraPixelArray) return false

  if (modulosConectados) {
    if (resolucion !== resolucionActual) {
      XR8.removeCameraPipelineModule('camerapixelarray')
      XR8.addCameraPipelineModule(
        XR8.CameraPixelArray.pipelineModule({luminance: false, maxDimension: resolucion}),
      )
      resolucionActual = resolucion
    }
    return true
  }

  XR8.addCameraPipelineModule(
    XR8.CameraPixelArray.pipelineModule({luminance: false, maxDimension: resolucion}),
  )
  XR8.addCameraPipelineModule({
    name: XR_MODULO,
    onProcessCpu: ({processGpuResult}: any) => {
      const cpa = processGpuResult && processGpuResult.camerapixelarray
      if (!cpa || !cpa.pixels) return
      const {rows, cols, rowBytes, pixels} = cpa
      let img = frameCompartido.img
      if (!img || img.width !== cols || img.height !== rows) {
        img = new ImageData(cols, rows)
        frameCompartido.img = img
      }
      const out = img.data
      const anchoBytes = cols * 4
      for (let y = 0; y < rows; y++) {
        const srcY = frameCompartido.voltear ? rows - 1 - y : y
        const ini = srcY * rowBytes
        out.set(pixels.subarray(ini, ini + anchoBytes), y * anchoBytes)
      }
      frameCompartido.id++
    },
  })
  modulosConectados = true
  resolucionActual = resolucion
  console.log('[segmentacion] Conectado al pipeline de cámara de 8th Wall', resolucion)
  return true
}

const desconectarXR8 = () => {
  const XR8 = (window as any).XR8
  if (!modulosConectados || !XR8 || !XR8.removeCameraPipelineModule) return
  XR8.removeCameraPipelineModule(XR_MODULO)
  XR8.removeCameraPipelineModule('camerapixelarray')
  modulosConectados = false
  resolucionActual = 0
  frameCompartido.img = null
}

// ---------- MediaPipe ----------

// La librería y los archivos WASM se cargan una sola vez y se reutilizan
let mediapipe: Promise<{mp: any, fileset: any}> | null = null
const cargarMediaPipe = () => {
  if (!mediapipe) {
    mediapipe = (async () => {
      const mp = await import(/* @vite-ignore */ /* webpackIgnore: true */ MP_BUNDLE)
      const fileset = await mp.FilesetResolver.forVisionTasks(MP_WASM)
      return {mp, fileset}
    })()
  }
  return mediapipe
}

const crearSegmentador = async (modelo: Modelo, preferido: Delegado) => {
  const {mp, fileset} = await cargarMediaPipe()
  const opciones = (delegate: Delegado) => ({
    baseOptions: {modelAssetPath: MODELOS[modelo], delegate},
    runningMode: 'VIDEO',
    outputCategoryMask: false,
    outputConfidenceMasks: true,
  })
  if (preferido === 'GPU') {
    try {
      const seg = await mp.ImageSegmenter.createFromOptions(fileset, opciones('GPU'))
      return {seg, delegado: 'GPU' as Delegado}
    } catch (e) {
      console.warn('[segmentacion] GPU no disponible, usando CPU', e)
    }
  }
  const seg = await mp.ImageSegmenter.createFromOptions(fileset, opciones('CPU'))
  return {seg, delegado: 'CPU' as Delegado}
}

// En algunos celulares (sobre todo Android) el modo GPU de MediaPipe devuelve máscaras
// vacías (todo en 0). Se comprueba en los primeros frames.
const mascaraValida = (masks: any[]): boolean => {
  const arrays = masks.map(m => m.getAsFloat32Array())
  const n = arrays[0].length
  const muestras = 64
  if (arrays.length === 1) {
    // Modelo de un solo canal (persona): válido si detecta a alguien en algún punto
    let max = 0
    for (let s = 0; s < muestras; s++) {
      const v = arrays[0][Math.floor(((s + 0.5) / muestras) * n)]
      if (Number.isFinite(v) && v > max) max = v
    }
    return max > 0.5
  }
  // Modelos de varias clases: en cada pixel las clases deben sumar ~1
  let suma = 0
  for (let s = 0; s < muestras; s++) {
    const p = Math.floor(((s + 0.5) / muestras) * n)
    for (let k = 0; k < arrays.length; k++) {
      const v = arrays[k][p]
      if (Number.isFinite(v)) suma += v
    }
  }
  const media = suma / muestras
  return media > 0.5 && media < 1.5
}

// Reduce el frame a ~256 px promediando bloques de 2×2 (reutiliza memoria)
const reducirFrame = (src: ImageData, destino: ImageData | null): ImageData => {
  const escala = LADO_MODELO / Math.max(src.width, src.height)
  if (escala >= 1) return src
  const w = Math.max(1, Math.round(src.width * escala))
  const h = Math.max(1, Math.round(src.height * escala))
  const dst = destino && destino.width === w && destino.height === h ? destino : new ImageData(w, h)
  const s = src.data
  const d = dst.data
  const sw = src.width
  const fx = src.width / w
  const fy = src.height / h
  for (let y = 0; y < h; y++) {
    const y0 = Math.floor(y * fy)
    const y1 = Math.min(src.height - 1, Math.floor((y + 0.5) * fy))
    for (let x = 0; x < w; x++) {
      const x0 = Math.floor(x * fx)
      const x1 = Math.min(sw - 1, Math.floor((x + 0.5) * fx))
      const a = (y0 * sw + x0) * 4
      const b = (y0 * sw + x1) * 4
      const c = (y1 * sw + x0) * 4
      const e = (y1 * sw + x1) * 4
      const o = (y * w + x) * 4
      d[o] = (s[a] + s[b] + s[c] + s[e]) >> 2
      d[o + 1] = (s[a + 1] + s[b + 1] + s[c + 1] + s[e + 1]) >> 2
      d[o + 2] = (s[a + 2] + s[b + 2] + s[c + 2] + s[e + 2]) >> 2
      d[o + 3] = 255
    }
  }
  return dst
}

const normalizarColor = (r: number, g: number, b: number) => {
  const max = Math.max(r, g, b)
  return max > 1 ? [r / 255, g / 255, b / 255] : [r, g, b]
}

// ---------- Shader ----------

const VERTEX = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`

const FRAGMENT = /* glsl */ `
  uniform sampler2D uMask;     // máscara del modelo (baja resolución)
  uniform sampler2D uCam;      // el MISMO frame que se segmentó (alta resolución)
  uniform sampler2D uMap;
  uniform bool uUseMap;
  uniform vec3 uColor;
  uniform float uFeather;
  uniform bool uInvert;
  uniform bool uMirror;
  uniform vec2 uScale;
  uniform vec2 uOffset;
  uniform vec2 uMaskTexel;
  uniform float uIntensity;
  uniform float uThreshold;
  uniform float uReady;
  // Pelo / Piel / Ropa
  uniform bool uClase;
  uniform sampler2D uPrior;      // forma del modelo (nítida), compensada por movimiento
  uniform sampler2D uPriorAncha; // la misma forma, ensanchada
  uniform bool uUsaPrior;
  uniform sampler2D uLut;        // colores aprendidos: color → probabilidad
  uniform vec2 uCamTexel;
  varying vec2 vUv;

  // Probabilidad de que un color sea la clase (misma agrupación YCbCr que en la CPU)
  float probColor(vec3 rgb) {
    vec3 c = rgb * 255.0;
    float Y = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
    float Cb = 128.0 - 0.1687 * c.r - 0.3313 * c.g + 0.5 * c.b;
    float Cr = 128.0 + 0.5 * c.r - 0.4187 * c.g - 0.0813 * c.b;
    float yb = clamp(floor(Y / 32.0), 0.0, 7.0);
    float cbb = clamp(floor(Cb / 8.0), 0.0, 31.0);
    float crb = clamp(floor(Cr / 8.0), 0.0, 31.0);
    return texture2D(uLut, vec2((yb * 32.0 + crb + 0.5) / 256.0, (cbb + 0.5) / 32.0)).r;
  }

  // Refinado LOCAL del borde. Alrededor del pixel se toman 16 muestras en dos anillos.
  // Las que la máscara marca claramente "dentro" y "fuera" son ejemplos de color de cada lado.
  // El pixel se asigna al lado cuyo ejemplo MÁS PARECIDO esté más cerca en color
  // (vecino más cercano, no promedio: funciona aunque un lado tenga varios colores,
  // p. ej. cara + pelo, o trenzas + camisa).
  // Lejos de bordes, o si el color no permite decidir, se respeta la máscara del modelo.
  float refinar(sampler2D tex, vec2 p, vec3 c0, float radio) {
    float m0 = texture2D(tex, p).r;

    // Optimización: si en 4 direcciones la máscara vale lo mismo, no hay borde cerca
    vec2 r = uMaskTexel * radio;
    float a1 = texture2D(tex, p + vec2(r.x, 0.0)).r;
    float a2 = texture2D(tex, p - vec2(r.x, 0.0)).r;
    float a3 = texture2D(tex, p + vec2(0.0, r.y)).r;
    float a4 = texture2D(tex, p - vec2(0.0, r.y)).r;
    float mn = min(min(min(a1, a2), min(a3, a4)), m0);
    float mx = max(max(max(a1, a2), max(a3, a4)), m0);
    if (mx - mn < 0.05) return m0;

    float df = 1e3;   // distancia de color al ejemplo "dentro" más parecido
    float db = 1e3;   // distancia de color al ejemplo "fuera" más parecido
    for (int k = 0; k < 16; k++) {
      float ang = float(k) * 0.3926991 + ((mod(float(k), 2.0) < 0.5) ? 0.0 : 0.19635);
      float rad = (mod(float(k), 2.0) < 0.5) ? radio : radio * 0.45;
      vec2 q = p + vec2(cos(ang), sin(ang)) * uMaskTexel * rad;
      float mk = texture2D(tex, q).r;
      vec3 dc = c0 - texture2D(uCam, q).rgb;
      float d = dot(dc, dc);
      if (mk > 0.8) df = min(df, d);
      if (mk < 0.2) db = min(db, d);
    }
    if (df > 999.0 || db > 999.0) return m0;
    float porColor = smoothstep(0.3, 0.7, db / max(df + db, 0.00001));
    // Certeza: cuánto más cerca está de un lado que del otro
    float certeza = abs(db - df) / (db + df + 0.0001);
    return mix(m0, porColor, smoothstep(0.15, 0.6, certeza));
  }

  void main() {
    // Pantalla -> coordenadas del frame (recorte tipo "cover")
    vec2 v = vUv * uScale + uOffset;
    if (uMirror) v.x = 1.0 - v.x;
    vec2 p = vec2(v.x, 1.0 - v.y);

    vec3 cam = texture2D(uCam, p).rgb;
    // Silueta de la persona (o máscara del objetivo), con borde refinado
    float m = refinar(uMask, p, cam, 10.0);
    // Pelo / Piel / Ropa: máscara del modelo pesado, también refinada contra la imagen actual
    if (uClase) {
      float q = 0.0;
      if (uUsaPrior) {
        // FORMA: máscara del modelo con el borde reajustado a la imagen actual.
        // Cerca de esa forma (versión ensanchada) se permite que el color la amplíe.
        float ps = max(refinar(uPrior, p, cam, 12.0), 0.6 * texture2D(uPriorAncha, p).r);
        ps = clamp(ps, 0.03, 0.95);
        // COLOR: probabilidad según los colores aprendidos (con 4 vecinos contra el ruido)
        vec2 d = uCamTexel * 1.5;
        float pc = probColor(cam) * 2.0
          + probColor(texture2D(uCam, p + vec2(d.x, 0.0)).rgb)
          + probColor(texture2D(uCam, p - vec2(d.x, 0.0)).rgb)
          + probColor(texture2D(uCam, p + vec2(0.0, d.y)).rgb)
          + probColor(texture2D(uCam, p - vec2(0.0, d.y)).rgb);
        pc = clamp(pc / 6.0, 0.02, 0.95);
        // Fusión en "odds": un color ambiguo (~0.5) deja decidir a la forma
        float odds = (pc / (1.0 - pc)) * (ps / (1.0 - ps));
        q = odds / (1.0 + odds);
      }
      // Mínimo con la silueta (no producto: dos probabilidades medias no deben anularse)
      m = min(m, q);
    }
    if (uInvert) m = 1.0 - m;

    float f = max(uFeather, 0.002);
    float c = clamp(0.5 + uThreshold, 0.05, 0.95);
    m = smoothstep(c - f, c + f, m);

    vec3 col = uUseMap ? texture2D(uMap, vUv).rgb : uColor;
    vec3 outCol = mix(cam, col, m * uIntensity);
    // Opaco: esta imagen sustituye a la cámara de 8th Wall (cuando ya hay máscara)
    gl_FragColor = vec4(outCol, uReady);
  }
`

// ---------- Estado ----------

type Estado = {
  mesh: any
  material: any
  maskTex: any
  maskData: Uint8Array
  maskW: number
  maskH: number
  camTex: any
  camData: Uint8Array
  priorTex: any
  priorData: Uint8Array
  anchaTex: any
  anchaData: Uint8Array
  lutTex: any
  lutData: Uint8Array
  lutVersion: number
  pequena: ImageData | null
  mapTex: any
  imagenCargada: string
  segmentador: any
  modelo: Modelo | null
  clasificador: Clasificador | null
  clasificadorDe: string
  delegado: Delegado
  forzarCPU: boolean      // la GPU de este dispositivo ya falló: siempre CPU
  creando: boolean
  validado: boolean
  framesInvalidos: number
  frame: number
  ultimoFrameId: number
  ultimoTs: number
  avisoMostrado: boolean
  msSegmentar: number
  medidas: number
  pos: any
  quat: any
}

// El eid de 8th Wall es un entero de 64 bits (bigint)
const instancias = new Map<bigint, Estado>()

const crearTextura = (w: number, h: number) => {
  const data = new Uint8Array(w * h * 4)
  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat)
  tex.minFilter = THREE.LinearFilter
  tex.magFilter = THREE.LinearFilter
  tex.needsUpdate = true
  return {data, tex}
}

// Crea (o cambia) el segmentador para un modelo; cierra el anterior
const prepararSegmentador = (eid: bigint, st: Estado, modelo: Modelo, preferido: Delegado) => {
  st.creando = true
  const viejo = st.segmentador
  st.segmentador = null
  if (viejo) {
    try { viejo.close() } catch (e) { /* nada */ }
  }
  crearSegmentador(modelo, preferido)
    .then(({seg, delegado}) => {
      if (!instancias.has(eid)) {
        seg.close()
        return
      }
      st.segmentador = seg
      st.modelo = modelo
      st.delegado = delegado
      st.validado = delegado === 'CPU'   // en CPU el resultado es confiable
      st.framesInvalidos = 0
      st.creando = false
      console.log('[segmentacion] MediaPipe listo —', modelo, '—', delegado)
    })
    .catch((e) => {
      st.creando = false
      console.error('[segmentacion] No se pudo cargar MediaPipe', e)
    })
}

// ---------- Componente ----------

ecs.registerComponent({
  name: 'Segmentacion',

  schema: {
    // @label Objetivo
    // @enum Fondo, Persona, Pelo, Piel, Ropa
    objetivo: ecs.string,

    // @label Usar imagen
    usarImagen: ecs.boolean,

    // @label Imagen
    // @asset
    // @condition usarImagen=true
    imagen: ecs.string,

    // @group start color:color
    // @group condition usarImagen=false
    colorR: ecs.f32,
    colorG: ecs.f32,
    colorB: ecs.f32,
    // @group end

    // @label Intensidad (1 = reemplazo total)
    // @min 0
    // @max 1
    intensidad: ecs.f32,

    // @label Suavizado del borde
    // @min 0
    // @max 0.2
    suavizado: ecs.f32,

    // @label Ajuste del borde (− expande / + contrae la persona)
    // @min -0.3
    // @max 0.3
    ajusteBorde: ecs.f32,

    // @label Procesamiento (Auto = GPU con respaldo en CPU)
    // @enum Auto, CPU
    procesamiento: ecs.string,

    // @label Calidad de imagen
    // @enum 480, 540, 720
    calidad: ecs.string,

    // @label Invertir
    invertir: ecs.boolean,

    // @label Espejo (horizontal)
    espejo: ecs.boolean,

    // @label Voltear vertical
    voltearVertical: ecs.boolean,

    // @label Procesar cada N frames
    // @min 1
    // @max 4
    cadaNFrames: ecs.ui8,
  },

  schemaDefaults: {
    objetivo: 'Fondo',
    usarImagen: false,
    colorR: 0.1,
    colorG: 0.0,
    colorB: 0.2,
    intensidad: 1,
    suavizado: 0.03,
    ajusteBorde: 0.05,
    procesamiento: 'Auto',
    calidad: '540',
    invertir: false,
    espejo: false,
    voltearVertical: false,
    cadaNFrames: 1,
  },

  add: (world, component) => {
    const {data: maskData, tex: maskTex} = crearTextura(1, 1)
    const {data: camData, tex: camTex} = crearTextura(1, 1)
    const {data: priorData, tex: priorTex} = crearTextura(1, 1)
    const {data: anchaData, tex: anchaTex} = crearTextura(1, 1)
    // Tabla color → probabilidad: se lee el valor exacto (sin interpolar)
    const {data: lutData, tex: lutTex} = crearTextura(256, 32)
    lutTex.minFilter = THREE.NearestFilter
    lutTex.magFilter = THREE.NearestFilter

    const material = new THREE.ShaderMaterial({
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      uniforms: {
        uMask: {value: maskTex},
        uCam: {value: camTex},
        uMap: {value: null},
        uUseMap: {value: false},
        uColor: {value: new THREE.Color(0, 0, 0)},
        uFeather: {value: 0.03},
        uInvert: {value: false},
        uMirror: {value: false},
        uScale: {value: new THREE.Vector2(1, 1)},
        uOffset: {value: new THREE.Vector2(0, 0)},
        uMaskTexel: {value: new THREE.Vector2(1 / 256, 1 / 256)},
        uIntensity: {value: 1},
        uThreshold: {value: 0.05},
        uReady: {value: 0},
        uClase: {value: false},
        uPrior: {value: priorTex},
        uUsaPrior: {value: false},
        uPriorAncha: {value: anchaTex},
        uLut: {value: lutTex},
        uCamTexel: {value: new THREE.Vector2(1 / 540, 1 / 720)},
      },
    })

    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material)
    mesh.frustumCulled = false
    mesh.renderOrder = -1000  // se dibuja antes que tus objetos 3D
    world.three.scene.add(mesh)

    const estado: Estado = {
      mesh,
      material,
      maskTex,
      maskData,
      maskW: 1,
      maskH: 1,
      camTex,
      camData,
      priorTex,
      priorData,
      anchaTex,
      anchaData,
      lutTex,
      lutData,
      lutVersion: -1,
      pequena: null,
      mapTex: null,
      imagenCargada: '',
      segmentador: null,
      modelo: null,
      clasificador: null,
      clasificadorDe: '',
      delegado: 'GPU',
      forzarCPU: false,
      creando: false,
      validado: false,
      framesInvalidos: 0,
      frame: 0,
      ultimoFrameId: -1,
      ultimoTs: 0,
      avisoMostrado: false,
      msSegmentar: 0,
      medidas: 0,
      pos: new THREE.Vector3(),
      quat: new THREE.Quaternion(),
    }
    instancias.set(component.eid, estado)
    console.log('[segmentacion] Versión', VERSION)
  },

  tick: (world, component) => {
    const st = instancias.get(component.eid)
    if (!st) return
    const s = component.schema
    const cam = world.three.activeCamera
    if (!cam) return
    const u = st.material.uniforms

    // 1) Plano a pantalla completa, lejos de la cámara (detrás de tus objetos 3D)
    const d = Math.min(cam.far * 0.9, 1000)
    const fov = cam.isPerspectiveCamera ? cam.fov : 60
    const alto = 2 * d * Math.tan(THREE.MathUtils.degToRad(fov / 2)) * 1.02
    const ancho = alto * (cam.aspect || 1)
    cam.getWorldPosition(st.pos)
    cam.getWorldQuaternion(st.quat)
    st.mesh.position.copy(st.pos)
    st.mesh.quaternion.copy(st.quat)
    st.mesh.translateZ(-d)
    st.mesh.scale.set(ancho, alto, 1)

    // 2) Parámetros del panel
    const objetivo = OBJETIVOS[s.objetivo] || OBJETIVOS.Fondo
    const [r, g, b] = normalizarColor(s.colorR, s.colorG, s.colorB)
    u.uColor.value.setRGB(r, g, b)
    u.uFeather.value = s.suavizado
    u.uInvert.value = s.invertir
    u.uMirror.value = s.espejo
    u.uIntensity.value = s.intensidad
    // "Ajuste del borde" siempre se refiere a la PERSONA (+ = persona más ajustada)
    const pintaFondo = objetivo.complemento !== !!s.invertir
    u.uThreshold.value = pintaFondo ? -s.ajusteBorde : s.ajusteBorde
    u.uClase.value = !!objetivo.clase
    frameCompartido.voltear = s.voltearVertical

    if (s.usarImagen && s.imagen && s.imagen !== st.imagenCargada) {
      st.imagenCargada = s.imagen
      new THREE.TextureLoader().load(s.imagen, (tex: any) => {
        if (st.mapTex) st.mapTex.dispose()
        st.mapTex = tex
        u.uMap.value = tex
      })
    }
    u.uUseMap.value = s.usarImagen && !!st.mapTex

    // 3) Modelo rápido (siempre) + modelo pesado en hilo aparte (solo Pelo / Piel / Ropa)
    if (!st.modelo && !st.creando) {
      const preferido: Delegado = (s.procesamiento === 'CPU' || st.forzarCPU) ? 'CPU' : 'GPU'
      prepararSegmentador(component.eid, st, 'selfie', preferido)
      st.modelo = 'selfie'
    }
    if (objetivo.clase) {
      if (st.clasificadorDe !== s.objetivo) {
        if (st.clasificador) st.clasificador.worker.terminate()
        st.clasificador = crearClasificador(objetivo.clase, objetivo.canales)
        st.clasificadorDe = s.objetivo
      }
    } else if (st.clasificador) {
      st.clasificador.worker.terminate()
      st.clasificador = null
      st.clasificadorDe = ''
    }

    // 4) Frame de cámara desde 8th Wall
    const resolucion = parseInt(s.calidad, 10) || 540
    if (!conectarXR8(resolucion)) {
      if (!st.avisoMostrado) {
        console.warn('[segmentacion] Esperando a XR8 (¿la cámara tiene AR activado?)')
        st.avisoMostrado = true
      }
      return
    }
    const img = frameCompartido.img
    if (!st.segmentador || !img || frameCompartido.id === st.ultimoFrameId) return

    // Cada N frames: imagen y máscara se actualizan JUNTAS para que nunca se desalineen
    st.frame++
    if (st.frame % Math.max(1, s.cadaNFrames) !== 0) return
    st.ultimoFrameId = frameCompartido.id

    let ts = performance.now()
    if (ts <= st.ultimoTs) ts = st.ultimoTs + 1
    st.ultimoTs = ts

    let mascaraOk = false
    try {
      // El modelo trabaja a 256 px: se le pasa una copia reducida del frame
      st.pequena = reducirFrame(img, st.pequena)
      const t0 = performance.now()

      // segmentForVideo es síncrono: la máscara corresponde a ESTE frame
      st.segmentador.segmentForVideo(st.pequena, ts, (result: any) => {
        const masks = result.confidenceMasks
        if (!masks || !masks.length) return

        if (!st.validado) {
          if (mascaraValida(masks)) {
            st.validado = true
            console.log('[segmentacion] Máscara verificada —', st.delegado)
          } else {
            st.framesInvalidos++
            return
          }
        }
        mascaraOk = true

        const w = masks[0].width
        const h = masks[0].height
        if (w !== st.maskW || h !== st.maskH) {
          st.maskTex.dispose()
          const nuevo = crearTextura(w, h)
          st.maskTex = nuevo.tex
          st.maskData = nuevo.data
          st.maskW = w
          st.maskH = h
          u.uMask.value = nuevo.tex
          const f = Math.max(w, h) / 256
          u.uMaskTexel.value.set(f / w, f / h)
        }

        // Persona del frame ACTUAL (modelo rápido)
        const persona = masks[0].getAsFloat32Array()
        const out = st.maskData
        const n = w * h
        const cl = st.clasificador

        if (objetivo.clase) {
          // Pelo / Piel / Ropa: aquí se preparan la silueta actual y la máscara del modelo
          // pesado (desplazada según el movimiento); el shader afina ambos bordes.
          const tieneMascara = !!(cl && cl.zona)

          // a) Silueta actual (límite contra el fondo)
          for (let p = 0; p < n; p++) {
            let v = persona[p]
            v = v < 0 ? 0 : v > 1 ? 1 : v
            const byte = tieneMascara ? (v * 255) | 0 : 0   // sin máscara de clase aún: no se pinta
            const o = p * 4
            out[o] = byte
            out[o + 1] = byte
            out[o + 2] = byte
            out[o + 3] = 255
          }

          // b) Colores aprendidos → tabla (solo cuando el modelo pesado respondió de nuevo)
          if (cl && cl.zona && cl.version !== st.lutVersion) {
            const L = st.lutData
            const prob = cl.probabilidad
            for (let yb = 0; yb < BY; yb++) {
              for (let cb = 0; cb < BC; cb++) {
                for (let cr = 0; cr < BC; cr++) {
                  const v = (prob[(yb << 10) | (cb << 5) | cr] * 255) | 0
                  const o = (cb * 256 + yb * 32 + cr) * 4
                  L[o] = v
                  L[o + 1] = v
                  L[o + 2] = v
                  L[o + 3] = 255
                }
              }
            }
            st.lutTex.needsUpdate = true
            st.lutVersion = cl.version
          }

          // c) Forma del modelo (nítida y ancha), desplazada/escalada según el movimiento
          const zona = cl && cl.zona && cl.zonaW === w && cl.zonaH === h ? cl.zona : null
          const ancha = cl && cl.zonaAncha && cl.zonaW === w && cl.zonaH === h ? cl.zonaAncha : null
          u.uUsaPrior.value = !!(zona && ancha)
          if (zona && ancha) {
            if (st.priorTex.image.width !== w || st.priorTex.image.height !== h) {
              st.priorTex.dispose()
              st.anchaTex.dispose()
              const t1 = crearTextura(w, h)
              const t2 = crearTextura(w, h)
              st.priorTex = t1.tex
              st.priorData = t1.data
              st.anchaTex = t2.tex
              st.anchaData = t2.data
              u.uPrior.value = t1.tex
              u.uPriorAncha.value = t2.tex
            }
            const antes = cl ? cl.zonaPos : null
            const ahora = antes ? medirSilueta(persona, w, h) : null
            const esc = antes && ahora ? antes.tam / ahora.tam : 1
            const P = st.priorData
            const A = st.anchaData
            for (let y = 0; y < h; y++) {
              for (let x = 0; x < w; x++) {
                let zp = y * w + x
                if (antes && ahora) {
                  const xs = Math.round(antes.cx + (x - ahora.cx) * esc)
                  const ys = Math.round(antes.cy + (y - ahora.cy) * esc)
                  zp = xs >= 0 && xs < w && ys >= 0 && ys < h ? ys * w + xs : -1
                }
                const v = zp >= 0 ? (zona[zp] * 255) | 0 : 0
                const va = zp >= 0 ? (ancha[zp] * 255) | 0 : 0
                const o = (y * w + x) * 4
                P[o] = v
                P[o + 1] = v
                P[o + 2] = v
                P[o + 3] = 255
                A[o] = va
                A[o + 1] = va
                A[o + 2] = va
                A[o + 3] = 255
              }
            }
            st.priorTex.needsUpdate = true
            st.anchaTex.needsUpdate = true
          }

          // El modelo pesado recibe este frame cuando esté libre (no bloquea este hilo)
          if (cl) enviarAClasificador(cl, st.pequena as ImageData, persona)
        } else {
          for (let p = 0; p < n; p++) {
            let v = persona[p]
            v = v < 0 ? 0 : v > 1 ? 1 : v
            if (objetivo.complemento) v = 1 - v
            const byte = (v * 255) | 0
            const o = p * 4
            out[o] = byte
            out[o + 1] = byte
            out[o + 2] = byte
            out[o + 3] = 255
          }
        }
        st.maskTex.needsUpdate = true
      })

      // Medición de rendimiento: promedio cada 120 segmentaciones
      st.msSegmentar += performance.now() - t0
      st.medidas++
      if (st.medidas === 120) {
        console.log('[segmentacion] Tiempo medio por frame:',
          (st.msSegmentar / 120).toFixed(1), 'ms —', st.modelo, '—', st.delegado)
        st.msSegmentar = 0
        st.medidas = 0
      }
    } catch (e) {
      console.warn('[segmentacion] Error segmentando frame', e)
      if (st.delegado === 'GPU') st.framesInvalidos++
    }

    // Si la GPU de este dispositivo no da máscaras válidas, se pasa a CPU (y se recuerda)
    if (!mascaraOk && st.delegado === 'GPU' && st.framesInvalidos >= 3 && !st.creando) {
      console.warn('[segmentacion] La GPU de este dispositivo no da máscaras válidas. Cambiando a CPU…')
      st.forzarCPU = true
      prepararSegmentador(component.eid, st, 'selfie', 'CPU')
    }
    if (!mascaraOk) return

    // 5) Copiar ESE MISMO frame a la textura que se muestra en pantalla
    if (st.camTex.image.width !== img.width || st.camTex.image.height !== img.height) {
      st.camTex.dispose()
      const nueva = crearTextura(img.width, img.height)
      st.camTex = nueva.tex
      st.camData = nueva.data
      u.uCam.value = nueva.tex
      u.uCamTexel.value.set(1 / img.width, 1 / img.height)
    }
    st.camData.set(img.data)
    st.camTex.needsUpdate = true

    // Recorte "cover": cómo encaja el frame en la pantalla
    const aspFrame = img.width / img.height
    const aspPantalla = cam.aspect || 1
    if (aspFrame > aspPantalla) {
      const sx = aspPantalla / aspFrame
      u.uScale.value.set(sx, 1)
      u.uOffset.value.set((1 - sx) / 2, 0)
    } else {
      const sy = aspFrame / aspPantalla
      u.uScale.value.set(1, sy)
      u.uOffset.value.set(0, (1 - sy) / 2)
    }

    if (u.uReady.value === 0) {
      console.log('[segmentacion] Primera máscara generada', st.maskW, 'x', st.maskH,
        '— imagen', img.width, 'x', img.height)
    }
    u.uReady.value = 1
  },

  remove: (world, component) => {
    const st = instancias.get(component.eid)
    if (!st) return
    world.three.scene.remove(st.mesh)
    st.mesh.geometry.dispose()
    st.material.dispose()
    st.maskTex.dispose()
    st.camTex.dispose()
    st.priorTex.dispose()
    st.anchaTex.dispose()
    st.lutTex.dispose()
    if (st.mapTex) st.mapTex.dispose()
    if (st.segmentador) st.segmentador.close()
    if (st.clasificador) st.clasificador.worker.terminate()
    instancias.delete(component.eid)
    if (instancias.size === 0) desconectarXR8()
  },
})