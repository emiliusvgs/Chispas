// segmentation.ts — Segmentación tipo Meta Spark para 8th Wall (Studio / Desktop)
// Objetivos: Fondo / Persona / Pelo / Piel / Ropa. Se pintan con un color o una imagen.
//
// Diseño (una sola ruta de calidad, elegida automáticamente según el celular):
//   · Motor principal: el modelo multiclase de MediaPipe (Google, Apache 2.0), convertido a
//     TensorFlow.js y ejecutado en la GPU. Da las 6 clases (fondo, pelo, piel del cuerpo,
//     piel de la cara, ropa, otros) en cada análisis, con la calidad del modelo completo.
//   · El modelo corre en un hilo aparte con su propia GPU, sin frenar la cámara ni tus 3D.
//     Usa WebGPU si el celular lo tiene (más rápido); si no, o si WebGL resulta más rápido
//     en ese celular, usa WebGL. Se mide al iniciar y se elige solo.
//   · Se ve la cámara en vivo de 8th Wall (sin retraso). Como el modelo tarda unos 70 ms, la
//     máscara se lleva a la posición ACTUAL con un flujo óptico por bloques (mide cuánto se
//     movió cada zona de la imagen desde el frame analizado) y el borde se afina con los
//     colores del frame actual.
//   · Respaldo automático: si el celular no puede usar la GPU con TF.js, se usa MediaPipe en
//     CPU (modelo selfie para Persona/Fondo y multiclase en un hilo aparte para el resto).
//
// El modelo va incluido en modelo-multiclase.ts (misma carpeta que este archivo): no depende
// de carpetas de assets, hosting ni permisos.

import * as ecs from '@8thwall/ecs'
import {GRAFO_MULTICLASE, PESOS_MULTICLASE_B64} from './modelo-multiclase'

const {THREE} = window as any

const VERSION = '18 (WebGPU + compensación de movimiento)'

// ---------- Archivos externos ----------
// Para producción: descarga estos archivos y sirve todo desde tu propio dominio.
const TFJS = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js'
const TFJS_WEBGPU = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgpu@4.22.0/dist/tf-backend-webgpu.min.js'
const MP_VERSION = '0.10.14'
const MP_BUNDLE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.mjs`
const MP_WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`
const MP_MODELOS = 'https://storage.googleapis.com/mediapipe-models/image_segmenter'
const MP_SELFIE = `${MP_MODELOS}/selfie_segmenter/float16/latest/selfie_segmenter.tflite`
const MP_MULTICLASE = `${MP_MODELOS}/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite`

// Resolución de la imagen que se muestra (lado más largo)
const RESOLUCION = 640
// Tamaño de entrada del modelo multiclase
const LADO = 256

// Canales del modelo multiclase: 0 fondo, 1 pelo, 2 piel cuerpo, 3 piel cara, 4 ropa, 5 otros
type Objetivo = {canales: number[], complemento: boolean}
const OBJETIVOS: Record<string, Objetivo> = {
  Fondo: {canales: [0], complemento: false},
  Persona: {canales: [0], complemento: true},   // persona = 1 − fondo
  Pelo: {canales: [1], complemento: false},
  Piel: {canales: [2, 3], complemento: false},
  Ropa: {canales: [4], complemento: false},
}

// ---------- Frames de cámara desde 8th Wall ----------

const XR_MODULO = 'segmentacion-frames'
// Último frame de la cámara (y un contador para saber cuándo llega uno nuevo)
const frameCompartido = {img: null as ImageData | null, id: 0}
let modulosConectados = false

const conectarXR8 = () => {
  if (modulosConectados) return true
  const XR8 = (window as any).XR8
  if (!XR8 || !XR8.addCameraPipelineModule || !XR8.CameraPixelArray) return false
  XR8.addCameraPipelineModule(XR8.CameraPixelArray.pipelineModule({luminance: false, maxDimension: RESOLUCION}))
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
        const ini = y * rowBytes
        out.set(pixels.subarray(ini, ini + anchoBytes), y * anchoBytes)
      }
      frameCompartido.id++
    },
  })
  modulosConectados = true
  console.log('[segmentacion] Conectado al pipeline de cámara de 8th Wall', RESOLUCION)
  return true
}

const desconectarXR8 = () => {
  const XR8 = (window as any).XR8
  if (!modulosConectados || !XR8 || !XR8.removeCameraPipelineModule) return
  XR8.removeCameraPipelineModule(XR_MODULO)
  XR8.removeCameraPipelineModule('camerapixelarray')
  modulosConectados = false
  frameCompartido.img = null
}

// Reduce el frame a ~256 px (bloques 2×2) para los modelos de MediaPipe en CPU
const reducirFrame = (src: ImageData, destino: ImageData | null): ImageData => {
  const escala = LADO / Math.max(src.width, src.height)
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

// ---------- Motor principal: modelo multiclase en TF.js (GPU) ----------

// Intérprete mínimo del modelo TFLite exportado (grafo JSON + pesos float16) sobre TF.js.
// Ejecuta en la GPU las 11 operaciones del modelo multiclase de MediaPipe.
const cargarModeloTF = (tf: any, grafo: any, pesosBuffer: ArrayBuffer) => {
  const f16 = new Uint16Array(pesosBuffer)
  const aF32 = (offset: number, n: number) => {
    const out = new Float32Array(n)
    const base = offset / 2
    for (let i = 0; i < n; i++) {
      const h = f16[base + i]
      const s = (h & 0x8000) ? -1 : 1
      const e = (h >> 10) & 0x1f
      const m = h & 0x3ff
      out[i] = e === 0 ? s * Math.pow(2, -14) * (m / 1024)
        : e === 31 ? (m ? NaN : s * Infinity)
          : s * Math.pow(2, e - 15) * (1 + m / 1024)
    }
    return out
  }
  const T = grafo.tensores
  const constantes = new Map<string, any>()
  const esConst = (i: number) => T[i].offset !== undefined
  const crear = (i: number, modo: string, mult = 1) => {
    const clave = i + ':' + modo
    if (constantes.has(clave)) return constantes.get(clave)
    const t = T[i]
    let x = tf.tensor(aF32(t.offset, t.n), t.forma, 'float32')
    if (modo === 'conv') x = tf.transpose(x, [1, 2, 3, 0])                    // OHWI → HWIO
    else if (modo === 'dw') x = tf.reshape(x, [t.forma[1], t.forma[2], t.forma[3] / mult, mult])
    else if (modo === 'tconv') x = tf.transpose(x, [1, 2, 0, 3])              // OHWI → HWOI
    x = tf.keep(x)
    constantes.set(clave, x)
    return x
  }
  const activar = (x: any, act: number) => {
    if (act === 0) return x
    if (act === 1) return tf.relu(x)
    if (act === 2) return tf.clipByValue(x, -1, 1)
    if (act === 3) return tf.relu6(x)
    if (act === 4) return tf.tanh(x)
    throw new Error('Activación no soportada: ' + act)
  }
  for (const op of grafo.ops) {
    if (op.tipo === 'CONV_2D') {
      crear(op.ent[1], 'conv')
      if (op.ent[2] >= 0) crear(op.ent[2], '')
    } else if (op.tipo === 'DEPTHWISE_CONV_2D') {
      crear(op.ent[1], 'dw', op.op.mult || 1)
      if (op.ent[2] >= 0) crear(op.ent[2], '')
    } else if (op.tipo === 'TRANSPOSE_CONV') {
      crear(op.ent[1], 'tconv')
      if (op.ent[3] >= 0) crear(op.ent[3], '')
    } else {
      for (const i of op.ent) if (i >= 0 && esConst(i)) crear(i, '')
    }
  }
  const predecir = (entrada: any) => tf.tidy(() => {
    const v = new Map<number, any>()
    v.set(grafo.entrada, entrada)
    const val = (i: number) => v.has(i) ? v.get(i) : constantes.get(i + ':')
    for (const op of grafo.ops) {
      const o = op.op
      const e = op.ent
      let r: any
      switch (op.tipo) {
        case 'CONV_2D':
          r = tf.conv2d(val(e[0]), constantes.get(e[1] + ':conv'), [o.sh, o.sw], o.padding, 'NHWC', [o.dh, o.dw])
          if (e[2] >= 0) r = tf.add(r, val(e[2]))
          r = activar(r, o.act)
          break
        case 'DEPTHWISE_CONV_2D':
          r = tf.depthwiseConv2d(val(e[0]), constantes.get(e[1] + ':dw'), [o.sh, o.sw], o.padding, 'NHWC', [o.dh, o.dw])
          if (e[2] >= 0) r = tf.add(r, val(e[2]))
          r = activar(r, o.act)
          break
        case 'TRANSPOSE_CONV':
          r = tf.conv2dTranspose(val(e[2]), constantes.get(e[1] + ':tconv'), T[e[0]].valor, [o.sh, o.sw], o.padding)
          if (e[3] !== undefined && e[3] >= 0) r = tf.add(r, val(e[3]))
          r = activar(r, o.act || 0)
          break
        case 'ADD': r = activar(tf.add(val(e[0]), val(e[1])), o.act); break
        case 'MUL': r = activar(tf.mul(val(e[0]), val(e[1])), o.act); break
        case 'RESHAPE': r = tf.reshape(val(e[0]), (e[1] !== undefined && T[e[1]].valor) ? T[e[1]].valor : o.shape); break
        case 'TRANSPOSE': r = tf.transpose(val(e[0]), T[e[1]].valor); break
        case 'SOFTMAX': r = tf.softmax(o.beta && o.beta !== 1 ? tf.mul(val(e[0]), o.beta) : val(e[0])); break
        case 'SUM': r = tf.sum(val(e[0]), T[e[1]].valor, o.keep); break
        case 'RESIZE_BILINEAR': r = tf.image.resizeBilinear(val(e[0]), T[e[1]].valor, o.align, o.half); break
        case 'RESIZE_NEAREST_NEIGHBOR': r = tf.image.resizeNearestNeighbor(val(e[0]), T[e[1]].valor, o.align, o.half); break
        default: throw new Error('Operación no soportada: ' + op.tipo)
      }
      v.set(op.sal[0], r)
    }
    return v.get(grafo.salida)
  })
  return {
    predecir,
    liberar: () => {
      for (const t of constantes.values()) t.dispose()
      constantes.clear()
    },
  }
}

const cargarScript = (url: string) => new Promise<void>((resolve, reject) => {
  const s = document.createElement('script')
  s.src = url
  s.onload = () => resolve()
  s.onerror = () => reject(new Error('No se pudo cargar ' + url))
  document.head.appendChild(s)
})

// Pesos del modelo incluidos en el proyecto (base64 → bytes)
const pesosIncluidos = (): ArrayBuffer => {
  const bin = atob(PESOS_MULTICLASE_B64)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  return bytes.buffer
}

type Motor = {
  nombre: string
  // Analiza un frame y devuelve la máscara del objetivo (0..1) y su tamaño
  analizar: (img: ImageData, objetivo: Objetivo) => Promise<{mascara: Float32Array, w: number, h: number}>
}

const crearMotorTF = async (): Promise<Motor> => {
  const w = window as any
  if (!w.tf) await cargarScript(TFJS)
  const tf = w.tf
  // Texturas de 16 bits: lo más rápido y compatible en las pruebas en celulares de gama media
  tf.env().set('WEBGL_FORCE_F16_TEXTURES', true)
  if (!(await tf.setBackend('webgl')) || tf.getBackend() !== 'webgl') throw new Error('TF.js no pudo usar la GPU')
  await tf.ready()
  const modelo = cargarModeloTF(tf, GRAFO_MULTICLASE, pesosIncluidos())

  // Verificación: en algunas GPUs los modelos devuelven ceros. Con una imagen de prueba, la
  // salida del modelo debe variar entre pixeles y clases (si todo es ~0, la GPU falla).
  const prueba = tf.tidy(() => {
    const salida = modelo.predecir(tf.randomUniform([1, LADO, LADO, 3], 0, 1))
    return tf.moments(salida).variance
  })
  const varianza = (await prueba.data())[0]
  prueba.dispose()
  if (!(varianza > 1e-3) || !Number.isFinite(varianza)) {
    modelo.liberar()
    throw new Error('la GPU devolvió resultados inválidos')
  }

  const analizar = async (img: ImageData, objetivo: Objetivo) => {
    const t = tf.tidy(() => {
      const x = tf.image.resizeBilinear(tf.browser.fromPixels(img).toFloat().div(255), [LADO, LADO], false, true)
      const prob = tf.softmax(modelo.predecir(tf.expandDims(x, 0)))          // [1,256,256,6]
      // Solo se lee la máscara del objetivo (65 mil valores en lugar de 393 mil)
      let m = tf.sum(tf.gather(prob, objetivo.canales, 3), 3)              // [1,256,256]
      if (objetivo.complemento) m = tf.sub(1, m)
      return m
    })
    const datos = await t.data()
    t.dispose()
    return {mascara: datos as Float32Array, w: LADO, h: LADO}
  }
  return {nombre: 'TF.js GPU (multiclase)', analizar}
}

// ---------- Motor principal: el mismo modelo en un hilo aparte (Web Worker) ----------
// El worker tiene su propio contexto de GPU (OffscreenCanvas). Así el modelo no compite con
// el render de 8th Wall en el hilo principal y la máscara llega en la mitad del tiempo.

const CODIGO_WORKER_TF = `
// Intérprete mínimo de modelos TFLite (exportados a JSON + pesos) sobre TensorFlow.js.
// Ejecuta en la GPU (backend WebGL de TF.js) las 11 operaciones que usa el modelo
// selfie_multiclass de MediaPipe.

function cargarModeloTF(tf, grafo, pesosBuffer) {
  const f16 = new Uint16Array(pesosBuffer)
  // float16 → float32
  const aF32 = (offset, n) => {
    const out = new Float32Array(n)
    const base = offset / 2
    for (let i = 0; i < n; i++) {
      const h = f16[base + i]
      const s = (h & 0x8000) ? -1 : 1
      const e = (h >> 10) & 0x1f
      const m = h & 0x3ff
      out[i] = e === 0 ? s * Math.pow(2, -14) * (m / 1024)
        : e === 31 ? (m ? NaN : s * Infinity)
          : s * Math.pow(2, e - 15) * (1 + m / 1024)
    }
    return out
  }

  const T = grafo.tensores
  const constantes = new Map()     // índice → tensor TF.js (ya en el formato que pide cada op)
  const enteros = (i) => T[i].valor

  const constante = (i, transformar) => {
    const clave = i + (transformar ? ':' + transformar : '')
    if (constantes.has(clave)) return constantes.get(clave)
    const t = T[i]
    let x = tf.tensor(aF32(t.offset, t.n), t.forma, 'float32')
    if (transformar === 'conv') x = tf.transpose(x, [1, 2, 3, 0])            // OHWI → HWIO
    else if (transformar === 'dw') {                                         // [1,H,W,C*m] → [H,W,C,m]
      x = tf.reshape(x, [t.forma[1], t.forma[2], t.forma[3] / constante.mult, constante.mult])
    } else if (transformar === 'tconv') x = tf.transpose(x, [1, 2, 0, 3])   // OHWI → HWOI
    x = tf.keep(x)
    constantes.set(clave, x)
    return x
  }

  const activar = (x, act) => {
    switch (act) {
      case 0: return x
      case 1: return tf.relu(x)
      case 2: return tf.clipByValue(x, -1, 1)
      case 3: return tf.relu6(x)
      case 4: return tf.tanh(x)
      default: throw new Error('Activación no soportada: ' + act)
    }
  }

  const esConst = (i) => T[i].offset !== undefined

  // Prepara todos los pesos una sola vez (transpuestos al formato de TF.js)
  for (const op of grafo.ops) {
    if (op.tipo === 'CONV_2D') { constante(op.ent[1], 'conv'); if (op.ent[2] >= 0) constante(op.ent[2]) }
    else if (op.tipo === 'DEPTHWISE_CONV_2D') {
      constante.mult = op.op.mult || 1
      constante(op.ent[1], 'dw'); if (op.ent[2] >= 0) constante(op.ent[2])
    } else if (op.tipo === 'TRANSPOSE_CONV') { constante(op.ent[1], 'tconv'); if (op.ent[3] >= 0) constante(op.ent[3]) }
    else for (const i of op.ent) if (i >= 0 && esConst(i)) constante(i)
  }

  const predecir = (entrada) => tf.tidy(() => {
    const v = new Map()
    v.set(grafo.entrada, entrada)
    const val = (i) => v.has(i) ? v.get(i) : constante(i)
    for (const op of grafo.ops) {
      const o = op.op
      const e = op.ent
      let r
      switch (op.tipo) {
        case 'CONV_2D': {
          r = tf.conv2d(val(e[0]), constantes.get(e[1] + ':conv'), [o.sh, o.sw], o.padding, 'NHWC', [o.dh, o.dw])
          if (e[2] >= 0) r = tf.add(r, constante(e[2]))
          r = activar(r, o.act)
          break
        }
        case 'DEPTHWISE_CONV_2D': {
          r = tf.depthwiseConv2d(val(e[0]), constantes.get(e[1] + ':dw'), [o.sh, o.sw], o.padding, 'NHWC', [o.dh, o.dw])
          if (e[2] >= 0) r = tf.add(r, constante(e[2]))
          r = activar(r, o.act)
          break
        }
        case 'TRANSPOSE_CONV': {
          const forma = enteros(e[0])
          r = tf.conv2dTranspose(val(e[2]), constantes.get(e[1] + ':tconv'), forma, [o.sh, o.sw], o.padding)
          if (e[3] !== undefined && e[3] >= 0) r = tf.add(r, constante(e[3]))
          r = activar(r, o.act || 0)
          break
        }
        case 'ADD': r = activar(tf.add(val(e[0]), val(e[1])), o.act); break
        case 'MUL': r = activar(tf.mul(val(e[0]), val(e[1])), o.act); break
        case 'RESHAPE': {
          const forma = (e[1] !== undefined && T[e[1]].valor) ? T[e[1]].valor : o.shape
          r = tf.reshape(val(e[0]), forma)
          break
        }
        case 'TRANSPOSE': r = tf.transpose(val(e[0]), enteros(e[1])); break
        case 'SOFTMAX': r = tf.softmax(o.beta && o.beta !== 1 ? tf.mul(val(e[0]), o.beta) : val(e[0])); break
        case 'SUM': r = tf.sum(val(e[0]), enteros(e[1]), o.keep); break
        case 'RESIZE_BILINEAR': r = tf.image.resizeBilinear(val(e[0]), enteros(e[1]), o.align, o.half); break
        case 'RESIZE_NEAREST_NEIGHBOR': r = tf.image.resizeNearestNeighbor(val(e[0]), enteros(e[1]), o.align, o.half); break
        default: throw new Error('Operación no soportada: ' + op.tipo)
      }
      v.set(op.sal[0], r)
    }
    return v.get(grafo.salida)
  })

  const liberar = () => { for (const t of constantes.values()) t.dispose(); constantes.clear() }
  return {predecir, liberar}
}

let tf = null
let modelo = null

const cargarJS = async (url) => {
  const txt = await (await fetch(url)).text()
  importScripts(URL.createObjectURL(new Blob([txt], {type: 'text/javascript'})))
}

// Prepara el modelo en un backend (webgpu o webgl), verifica que dé resultados válidos
// (algunas GPUs devuelven ceros) y mide cuánto tarda. Si algo falla devuelve null.
const probarBackend = async (nombre, m) => {
  let mod = null
  try {
    if (!(await tf.setBackend(nombre)) || tf.getBackend() !== nombre) return null
    await tf.ready()
    mod = cargarModeloTF(tf, m.grafo, m.pesos)
    const correr = async () => {
      const p = tf.tidy(() => tf.moments(mod.predecir(tf.randomUniform([1, m.lado, m.lado, 3], 0, 1))).variance)
      const v = (await p.data())[0]
      p.dispose()
      return v
    }
    const v = await correr()   // la primera vez compila los shaders
    if (!(v > 1e-3) || !Number.isFinite(v)) throw new Error('resultados inválidos')
    await correr()
    const t0 = performance.now()
    for (let i = 0; i < 3; i++) await correr()
    return {mod, ms: (performance.now() - t0) / 3}
  } catch (err) {
    if (mod) mod.liberar()
    return null
  }
}

self.onmessage = async (e) => {
  const m = e.data
  try {
    if (m.type === 'init') {
      await cargarJS(m.tfjs)
      tf = self.tf
      // 1) WebGPU: la vía moderna y más rápida, activa de fábrica en Chrome para Android
      let webgpu = null
      if (self.navigator && self.navigator.gpu) {
        try {
          await cargarJS(m.webgpu)
          webgpu = await probarBackend('webgpu', m)
        } catch (err) { webgpu = null }
      }
      // 2) WebGL: funciona en todos los celulares; se mide igual y se queda el más rápido
      if (m.software) tf.env().set('SOFTWARE_WEBGL_ENABLED', true)
      tf.env().set('WEBGL_FORCE_F16_TEXTURES', true)
      const webgl = await probarBackend('webgl', m)
      if (!webgpu && !webgl) throw new Error('la GPU no dio resultados válidos')
      const usarWebGPU = !!webgpu && (!webgl || webgpu.ms <= webgl.ms)
      const elegido = usarWebGPU ? webgpu : webgl
      const descartado = usarWebGPU ? webgl : webgpu
      if (descartado) descartado.mod.liberar()
      await tf.setBackend(usarWebGPU ? 'webgpu' : 'webgl')
      modelo = elegido.mod
      self.postMessage({type: 'listo', backend: usarWebGPU ? 'WebGPU' : 'WebGL',
        msWebGPU: webgpu ? webgpu.ms : null, msWebGL: webgl ? webgl.ms : null})
    } else if (m.type === 'frame') {
      const t = tf.tidy(() => {
        const x = tf.browser.fromPixels({data: new Uint8Array(m.data), width: m.lado, height: m.lado}, 3).toFloat().div(255)
        const prob = tf.softmax(modelo.predecir(tf.expandDims(x, 0)))
        let r = tf.sum(tf.gather(prob, m.canales, 3), 3)
        if (m.complemento) r = tf.sub(1, r)
        return r
      })
      const datos = await t.data()
      t.dispose()
      self.postMessage({type: 'mascara', id: m.id, mask: datos}, [datos.buffer])
    }
  } catch (err) {
    self.postMessage({type: 'error', id: m.id, msg: String((err && err.message) || err)})
  }
}
`

// Estira el frame a 256×256 (como espera el modelo) con interpolación bilineal
const estirarFrame = (src: ImageData): Uint8ClampedArray => {
  const out = new Uint8ClampedArray(LADO * LADO * 4)
  const s = src.data
  const sw = src.width
  const sh = src.height
  const fx = sw / LADO
  const fy = sh / LADO
  for (let y = 0; y < LADO; y++) {
    const yy = Math.min(sh - 1, Math.max(0, (y + 0.5) * fy - 0.5))
    const y0 = Math.floor(yy)
    const y1 = Math.min(sh - 1, y0 + 1)
    const wy = yy - y0
    for (let x = 0; x < LADO; x++) {
      const xx = Math.min(sw - 1, Math.max(0, (x + 0.5) * fx - 0.5))
      const x0 = Math.floor(xx)
      const x1 = Math.min(sw - 1, x0 + 1)
      const wx = xx - x0
      const a = (y0 * sw + x0) * 4
      const b = (y0 * sw + x1) * 4
      const c = (y1 * sw + x0) * 4
      const d = (y1 * sw + x1) * 4
      const o = (y * LADO + x) * 4
      for (let k = 0; k < 3; k++) {
        const arriba = s[a + k] + (s[b + k] - s[a + k]) * wx
        const abajo = s[c + k] + (s[d + k] - s[c + k]) * wx
        out[o + k] = arriba + (abajo - arriba) * wy
      }
      out[o + 3] = 255
    }
  }
  return out
}

const crearMotorTFWorker = async (): Promise<Motor> => {
  if (typeof Worker === 'undefined' || typeof OffscreenCanvas === 'undefined') throw new Error('sin OffscreenCanvas')
  const wk = new Worker(URL.createObjectURL(new Blob([CODIGO_WORKER_TF], {type: 'text/javascript'})))
  let idPedido = 0
  const pendientes = new Map<number, {res: (v: any) => void, rej: (e: Error) => void}>()
  let backend = 'WebGL'
  const listo = new Promise<void>((resolve, reject) => {
    const limite = setTimeout(() => reject(new Error('el hilo aparte no respondió')), 45000)
    wk.onmessage = (e: MessageEvent) => {
      const m = e.data
      const p = m.id !== undefined ? pendientes.get(m.id) : undefined
      if (m.type === 'listo') {
        clearTimeout(limite)
        backend = m.backend
        const f = (v: number | null) => v === null ? 'no disponible' : v.toFixed(1) + ' ms'
        console.log('[segmentacion] GPU medida — WebGPU:', f(m.msWebGPU), '· WebGL:', f(m.msWebGL), '→ se usa', m.backend)
        resolve()
      }
      else if (m.type === 'mascara' && p) { p.res(m); pendientes.delete(m.id) }
      else if (m.type === 'error') {
        if (p) { p.rej(new Error(m.msg)); pendientes.delete(m.id) } else { clearTimeout(limite); reject(new Error(m.msg)) }
      }
    }
    wk.onerror = (e: ErrorEvent) => { clearTimeout(limite); reject(new Error(e.message || 'error en el hilo aparte')) }
  })
  const pesos = pesosIncluidos()
  wk.postMessage({type: 'init', tfjs: TFJS, webgpu: TFJS_WEBGPU, grafo: GRAFO_MULTICLASE, pesos, lado: LADO,
    software: !!(window as any).__segSoftwareWebGL}, [pesos])
  try {
    await listo
  } catch (e) {
    wk.terminate()
    throw e
  }

  const analizar = async (img: ImageData, objetivo: Objetivo) => {
    const id = ++idPedido
    const datos = estirarFrame(img).buffer
    const r: any = await new Promise((res, rej) => {
      pendientes.set(id, {res, rej})
      wk.postMessage({type: 'frame', id, data: datos, lado: LADO,
        canales: objetivo.canales, complemento: objetivo.complemento}, [datos])
    })
    return {mascara: r.mask as Float32Array, w: LADO, h: LADO}
  }
  return {nombre: 'TF.js ' + backend + ' (hilo aparte)', analizar}
}

// ---------- Motor de respaldo: MediaPipe en CPU ----------

// Hilo aparte con el modelo multiclase en CPU (Pelo / Piel / Ropa sin frenar la cámara)
const CODIGO_WORKER = `
  let seg = null
  self.onmessage = async (e) => {
    const m = e.data
    try {
      if (m.type === 'init') {
        // El CDN sirve el bundle con un tipo que importScripts rechaza: se pasa por un blob
        const txt = await (await fetch(m.bundle)).text()
        self.exports = {}
        self.module = {exports: self.exports}
        importScripts(URL.createObjectURL(new Blob([txt], {type: 'text/javascript'})))
        const vision = self.module.exports
        const fileset = await vision.FilesetResolver.forVisionTasks(m.wasm)
        seg = await vision.ImageSegmenter.createFromOptions(fileset, {
          baseOptions: {modelAssetPath: m.model, delegate: 'CPU'},
          runningMode: 'IMAGE', outputCategoryMask: false, outputConfidenceMasks: true,
        })
        self.postMessage({type: 'listo'})
      } else if (m.type === 'frame' && seg) {
        const img = new ImageData(new Uint8ClampedArray(m.data), m.w, m.h)
        const res = seg.segment(img)
        const ms = res.confidenceMasks
        const w = ms[0].width, h = ms[0].height
        const arrs = m.canales.map((i) => ms[i].getAsFloat32Array())
        const out = new Float32Array(w * h)
        for (let p = 0; p < w * h; p++) {
          let v = 0
          for (let k = 0; k < arrs.length; k++) v += arrs[k][p]
          out[p] = m.complemento ? 1 - v : v
        }
        if (res.close) res.close()
        self.postMessage({type: 'mascara', id: m.id, w, h, mask: out}, [out.buffer])
      }
    } catch (err) {
      self.postMessage({type: 'error', id: m.id, msg: String((err && err.message) || err)})
    }
  }
`

const crearMotorCPU = async (): Promise<Motor> => {
  const mp = await import(/* @vite-ignore */ /* webpackIgnore: true */ MP_BUNDLE)
  const fileset = await mp.FilesetResolver.forVisionTasks(MP_WASM)
  // Persona / Fondo: modelo selfie, rápido en cualquier CPU
  const selfie = await mp.ImageSegmenter.createFromOptions(fileset, {
    baseOptions: {modelAssetPath: MP_SELFIE, delegate: 'CPU'},
    runningMode: 'IMAGE', outputCategoryMask: false, outputConfidenceMasks: true,
  })
  // Pelo / Piel / Ropa: multiclase en un hilo aparte
  let worker: Worker | null = null
  let listo: Promise<void> | null = null
  let idPedido = 0
  const pendientes = new Map<number, {res: (v: any) => void, rej: (e: Error) => void}>()
  const asegurarWorker = () => {
    if (listo) return listo
    const wk = new Worker(URL.createObjectURL(new Blob([CODIGO_WORKER], {type: 'text/javascript'})))
    worker = wk
    listo = new Promise<void>((resolve, reject) => {
      wk.onmessage = (e: MessageEvent) => {
        const m = e.data
        const p = m.id !== undefined ? pendientes.get(m.id) : undefined
        if (m.type === 'listo') resolve()
        else if (m.type === 'mascara' && p) { p.res(m); pendientes.delete(m.id) }
        else if (m.type === 'error') {
          if (p) { p.rej(new Error(m.msg)); pendientes.delete(m.id) } else reject(new Error(m.msg))
        }
      }
    })
    wk.postMessage({type: 'init', bundle: MP_BUNDLE.replace('vision_bundle.mjs', 'vision_bundle.cjs'),
      wasm: MP_WASM, model: MP_MULTICLASE})
    return listo
  }
  let pequena: ImageData | null = null

  const analizar = async (img: ImageData, objetivo: Objetivo) => {
    const peq = reducirFrame(img, pequena)
    pequena = peq
    if (objetivo.canales.length === 1 && objetivo.canales[0] === 0) {
      // Fondo / Persona con el modelo selfie (da "persona")
      const res = selfie.segment(peq)
      const mk = res.confidenceMasks[0]
      const persona = mk.getAsFloat32Array()
      const out = new Float32Array(persona.length)
      for (let p = 0; p < out.length; p++) out[p] = objetivo.complemento ? persona[p] : 1 - persona[p]
      const r = {mascara: out, w: mk.width, h: mk.height}
      if (res.close) res.close()
      return r
    }
    await asegurarWorker()
    const id = ++idPedido
    const copia = peq.data.slice().buffer
    const r: any = await new Promise((res, rej) => {
      pendientes.set(id, {res, rej})
      worker!.postMessage({type: 'frame', id, w: peq.width, h: peq.height, data: copia,
        canales: objetivo.canales, complemento: objetivo.complemento}, [copia])
    })
    return {mascara: r.mask, w: r.w, h: r.h}
  }
  return {nombre: 'MediaPipe CPU (respaldo)', analizar}
}

// Se elige el motor una sola vez por sesión: GPU en hilo aparte → GPU en hilo principal → CPU
let motorElegido: Promise<Motor> | null = null
const elegirMotor = () => {
  if (!motorElegido) {
    motorElegido = crearMotorTFWorker()
      .catch((e) => {
        console.warn('[segmentacion] Hilo aparte no disponible (' + (e && e.message) + '). Usando el hilo principal.')
        return crearMotorTF()
      })
      .catch((e) => {
        console.warn('[segmentacion] GPU con TF.js no disponible (' + (e && e.message) + '). Usando CPU.')
        return crearMotorCPU()
      })
    motorElegido
      .then(m => console.log('[segmentacion] Motor:', m.nombre))
      .catch(e => console.error('[segmentacion] No se pudo iniciar ningún motor', e))
  }
  return motorElegido
}

const normalizarColor = (r: number, g: number, b: number) => {
  const max = Math.max(r, g, b)
  return max > 1 ? [r / 255, g / 255, b / 255] : [r, g, b]
}

// ---------- Movimiento entre frames (flujo óptico por bloques) ----------
// El modelo tarda unos 70 ms en un celular de gama media: cuando la máscara llega, la persona
// ya se movió. Aquí se mide, bloque por bloque, cuánto se movió cada zona de la imagen entre
// el frame analizado y el frame actual, y el shader lleva la máscara a donde está todo AHORA.
// Funciona para cualquier movimiento (cabeza, cuerpo, ropa, pelo, la cámara) y corre en su
// propio hilo, así que no le quita tiempo a la cámara ni a tus objetos 3D.

const GRIS_LADO = 128   // lado largo de la imagen en gris que se compara
const BLOQUE = 8        // tamaño de cada bloque (en la imagen gris)

type Gris = {fino: Uint8Array, grueso: Uint8Array, w: number, h: number}

// Imagen en gris (lado largo 128) y su versión a la mitad, para buscar de grueso a fino
const aGris = (img: ImageData, destino: Gris | null): Gris => {
  const esc = GRIS_LADO / Math.max(img.width, img.height)
  const bw = Math.max(2, Math.round(img.width * esc / BLOQUE))
  const bh = Math.max(2, Math.round(img.height * esc / BLOQUE))
  const w = bw * BLOQUE
  const h = bh * BLOQUE
  const g = destino && destino.w === w && destino.h === h ? destino
    : {fino: new Uint8Array(w * h), grueso: new Uint8Array((w >> 1) * (h >> 1)), w, h}
  const s = img.data
  const sw = img.width
  const fx = img.width / w
  const fy = img.height / h
  for (let y = 0; y < h; y++) {
    const y0 = Math.min(img.height - 1, Math.floor((y + 0.25) * fy))
    const y1 = Math.min(img.height - 1, Math.floor((y + 0.75) * fy))
    for (let x = 0; x < w; x++) {
      const x0 = Math.min(sw - 1, Math.floor((x + 0.25) * fx))
      const x1 = Math.min(sw - 1, Math.floor((x + 0.75) * fx))
      const a = (y0 * sw + x0) * 4
      const b = (y0 * sw + x1) * 4
      const c = (y1 * sw + x0) * 4
      const d = (y1 * sw + x1) * 4
      g.fino[y * w + x] = (s[a] + 2 * s[a + 1] + s[a + 2] + s[b] + 2 * s[b + 1] + s[b + 2] +
        s[c] + 2 * s[c + 1] + s[c + 2] + s[d] + 2 * s[d + 1] + s[d + 2]) >> 4
    }
  }
  const gw = w >> 1
  const gh = h >> 1
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      const o = 2 * y * w + 2 * x
      g.grueso[y * gw + x] = (g.fino[o] + g.fino[o + 1] + g.fino[o + w] + g.fino[o + w + 1]) >> 2
    }
  }
  return g
}

// Hilo del flujo óptico. Búsqueda gruesa de ±7 px a media resolución (≈ ±70 px en la imagen)
// y ajuste fino con precisión de fracción de pixel.
const CODIGO_FLUJO = `
const BLOQUE = 8
const BUSQUEDA = 7
// Diferencia entre un bloque del frame actual y un bloque desplazado del frame de referencia
// (se corta en cuanto supera "limite": ya no puede ser el mejor)
const diferencia = (a, b, w, x, y, dx, dy, lado, limite) => {
  let suma = 0
  for (let j = 0; j < lado; j++) {
    if (suma >= limite) return suma
    let ia = (y + j) * w + x
    let ib = (y + j + dy) * w + x + dx
    for (let i = 0; i < lado; i++) {
      const d = a[ia++] - b[ib++]
      suma += d < 0 ? -d : d
    }
  }
  return suma
}

// Para cada bloque del frame ACTUAL: dónde estaba en el frame de REFERENCIA (en px de la imagen gris)
const costos = new Float64Array(25)
const calcularFlujo = (act, ref, sal, tmp, valido) => {
  const w = act.w
  const bw = act.w / BLOQUE
  const bh = act.h / BLOQUE
  const gw = w >> 1
  const gh = act.h >> 1
  const lg = BLOQUE >> 1
  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      const k = by * bw + bx
      // 1) Grueso (media resolución): búsqueda amplia, con una leve preferencia por "no se movió"
      const x = bx * lg
      const y = by * lg
      let textura = 0
      for (let j = 0; j < lg; j++) {
        for (let i = 0; i < lg; i++) {
          const o = (y + j) * gw + x + i
          if (i + 1 < lg) textura += Math.abs(act.grueso[o] - act.grueso[o + 1])
          if (j + 1 < lg) textura += Math.abs(act.grueso[o] - act.grueso[o + gw])
        }
      }
      // Zonas lisas (pared, tela sin textura) no dicen cuánto se movieron: se completan con sus vecinas
      valido[k] = textura > 3 * lg * lg ? 1 : 0
      tmp[2 * k] = 0
      tmp[2 * k + 1] = 0
      if (!valido[k]) continue
      let mejor = diferencia(act.grueso, ref.grueso, gw, x, y, 0, 0, lg, Infinity)
      let mx = 0
      let my = 0
      const x0 = Math.max(-BUSQUEDA, -x)
      const x1 = Math.min(BUSQUEDA, gw - lg - x)
      const y0 = Math.max(-BUSQUEDA, -y)
      const y1 = Math.min(BUSQUEDA, gh - lg - y)
      for (let dy = y0; dy <= y1; dy++) {
        for (let dx = x0; dx <= x1; dx++) {
          const pena = Math.abs(dx) + Math.abs(dy)
          if (pena >= mejor) continue
          const c = diferencia(act.grueso, ref.grueso, gw, x, y, dx, dy, lg, mejor - pena) + pena
          if (c < mejor) { mejor = c; mx = dx; my = dy }
        }
      }
      // 2) Fino (resolución completa): ajuste de ±2 px alrededor del resultado grueso
      //    y medio pixel más con una parábola sobre las diferencias vecinas
      const fx = bx * BLOQUE
      const fy = by * BLOQUE
      const cx = 2 * mx
      const cy = 2 * my
      let fmejor = Infinity
      let fmx = cx
      let fmy = cy
      for (let j = 0; j < 25; j++) costos[j] = Infinity
      for (let dy = cy - 2; dy <= cy + 2; dy++) {
        if (fy + dy < 0 || fy + dy + BLOQUE > act.h) continue
        for (let dx = cx - 2; dx <= cx + 2; dx++) {
          if (fx + dx < 0 || fx + dx + BLOQUE > w) continue
          const c = diferencia(act.fino, ref.fino, w, fx, fy, dx, dy, BLOQUE, Infinity)
          costos[(dy - cy + 2) * 5 + dx - cx + 2] = c
          if (c < fmejor) { fmejor = c; fmx = dx; fmy = dy }
        }
      }
      const ix = fmx - cx + 2
      const iy = fmy - cy + 2
      let sx = 0
      let sy = 0
      if (ix > 0 && ix < 4) {
        const l = costos[iy * 5 + ix - 1]
        const r = costos[iy * 5 + ix + 1]
        const den = l + r - 2 * fmejor
        if (Number.isFinite(den) && den > 0) sx = Math.max(-0.5, Math.min(0.5, (l - r) / (2 * den)))
      }
      if (iy > 0 && iy < 4) {
        const t = costos[(iy - 1) * 5 + ix]
        const bo = costos[(iy + 1) * 5 + ix]
        const den = t + bo - 2 * fmejor
        if (Number.isFinite(den) && den > 0) sy = Math.max(-0.5, Math.min(0.5, (t - bo) / (2 * den)))
      }
      tmp[2 * k] = fmx + sx
      tmp[2 * k + 1] = fmy + sy
    }
  }
  // Completar zonas lisas con el promedio de las vecinas válidas (de afuera hacia adentro)
  for (let pasada = 0, faltan = 1; faltan && pasada < bw + bh; pasada++) {
    faltan = 0
    for (let by = 0; by < bh; by++) {
      for (let bx = 0; bx < bw; bx++) {
        const k = by * bw + bx
        if (valido[k]) continue
        let sx = 0
        let sy = 0
        let n = 0
        for (let j = -1; j <= 1; j++) {
          for (let i = -1; i <= 1; i++) {
            const xx = bx + i
            const yy = by + j
            if (xx < 0 || yy < 0 || xx >= bw || yy >= bh) continue
            const v = yy * bw + xx
            if (valido[v] === 1) { sx += tmp[2 * v]; sy += tmp[2 * v + 1]; n++ }
          }
        }
        if (n) { tmp[2 * k] = sx / n; tmp[2 * k + 1] = sy / n; valido[k] = 2 } else faltan++
      }
    }
    for (let k = 0; k < bw * bh; k++) if (valido[k] === 2) valido[k] = 1
  }
  // Mediana 3×3: elimina bloques sueltos que se equivocaron
  const xs = []
  const ys = []
  let total = 0
  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      xs.length = 0
      ys.length = 0
      for (let j = -1; j <= 1; j++) {
        for (let i = -1; i <= 1; i++) {
          const xx = bx + i
          const yy = by + j
          if (xx < 0 || yy < 0 || xx >= bw || yy >= bh) continue
          const v = yy * bw + xx
          const ok = valido[v] === 1
          xs.push(ok ? tmp[2 * v] : 0)
          ys.push(ok ? tmp[2 * v + 1] : 0)
        }
      }
      xs.sort((a, b) => a - b)
      ys.sort((a, b) => a - b)
      const k = by * bw + bx
      sal[2 * k] = xs[xs.length >> 1]
      sal[2 * k + 1] = ys[ys.length >> 1]
      total += Math.abs(sal[2 * k]) + Math.abs(sal[2 * k + 1])
    }
  }
  return total / (bw * bh)   // movimiento medio (px de la imagen gris)
}


let referencia = null
let version = -1
self.onmessage = (e) => {
  const m = e.data
  if (m.ref) { referencia = m.ref; version = m.version }
  const act = m.act
  if (!referencia || referencia.w !== act.w || referencia.h !== act.h) { self.postMessage({version: -1}); return }
  const n = (act.w / BLOQUE) * (act.h / BLOQUE)
  const sal = new Float32Array(2 * n)
  const movimiento = calcularFlujo(act, referencia, sal, new Float32Array(2 * n), new Uint8Array(n))
  self.postMessage({version, flujo: sal, movimiento, w: act.w, h: act.h}, [sal.buffer])
}
`

// Sube el flujo como textura pequeña (un texel por bloque), en unidades de la imagen (0..1)
const RANGO_FLUJO = 0.25
const subirFlujo = (st: Estado, u: any, flujo: Float32Array, gw: number, gh: number) => {
  const bw = gw / BLOQUE
  const bh = gh / BLOQUE
  if (st.flowTex.image.width !== bw || st.flowTex.image.height !== bh) {
    st.flowTex.dispose()
    const nueva = crearTextura(bw, bh)
    st.flowTex = nueva.tex
    st.flowData = nueva.data
    u.uFlow.value = nueva.tex
  }
  const d = st.flowData
  for (let k = 0; k < bw * bh; k++) {
    const fx = flujo[2 * k] / gw / RANGO_FLUJO
    const fy = flujo[2 * k + 1] / gh / RANGO_FLUJO
    d[4 * k] = Math.max(0, Math.min(255, Math.round(fx * 127 + 128)))
    d[4 * k + 1] = Math.max(0, Math.min(255, Math.round(fy * 127 + 128)))
    d[4 * k + 2] = 0
    d[4 * k + 3] = 255
  }
  st.flowTex.needsUpdate = true
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
  uniform sampler2D uMask;     // máscara del objetivo (resolución del modelo)
  uniform sampler2D uCam;      // frame actual de la cámara (colores para afinar el borde)
  uniform sampler2D uFlow;     // movimiento de cada zona: frame actual → frame analizado
  uniform sampler2D uMap;
  uniform bool uUseMap;
  uniform vec3 uColor;
  uniform float uFeather;
  uniform bool uInvert;
  uniform vec2 uScale;
  uniform vec2 uOffset;
  uniform vec2 uMaskTexel;
  uniform float uIntensity;
  uniform float uThreshold;
  uniform float uReady;
  varying vec2 vUv;

  vec2 aFrame(vec2 q) {
    vec2 v = q * uScale + uOffset;
    return vec2(v.x, 1.0 - v.y);
  }

  // Afinado del borde: la máscara del modelo tiene menos resolución que la imagen.
  // Solo en la franja donde el modelo duda, cada pixel se asigna al lado cuyo ejemplo
  // de color más parecido esté más cerca (muestras de ambos lados en dos anillos).
  // Fuera de esa franja se respeta el modelo tal cual: no se abren huecos.
  float afinar(vec2 pm, vec2 p, vec3 c0) {
    float m0 = texture2D(uMask, pm).r;
    float franja = smoothstep(0.03, 0.25, m0) * (1.0 - smoothstep(0.75, 0.97, m0));
    if (franja < 0.01) return m0;
    float df = 1e3;
    float db = 1e3;
    for (int k = 0; k < 16; k++) {
      float ang = float(k) * 0.3926991 + ((mod(float(k), 2.0) < 0.5) ? 0.0 : 0.19635);
      float rad = (mod(float(k), 2.0) < 0.5) ? 4.0 : 2.0;
      vec2 o = vec2(cos(ang), sin(ang)) * uMaskTexel * rad;
      float mk = texture2D(uMask, pm + o).r;
      vec3 dc = c0 - texture2D(uCam, p + o).rgb;
      float d = dot(dc, dc);
      if (mk > 0.8) df = min(df, d);
      if (mk < 0.2) db = min(db, d);
    }
    if (df > 999.0 || db > 999.0) return m0;
    float porColor = smoothstep(0.3, 0.7, db / max(df + db, 0.00001));
    float certeza = abs(db - df) / (db + df + 0.0001);
    return mix(m0, porColor, smoothstep(0.25, 0.75, certeza) * franja);
  }

  void main() {
    // Pantalla -> coordenadas del frame (recorte tipo "cover")
    vec2 p = aFrame(vUv);
    // Dónde estaba este punto cuando se analizó la máscara (flujo óptico)
    vec2 fl = (texture2D(uFlow, p).rg * 255.0 - 128.0) / 127.0 * 0.25;
    vec2 pm = p + fl;
    vec3 cam = texture2D(uCam, p).rgb;
    float m = afinar(pm, p, cam);
    if (uInvert) m = 1.0 - m;
    float f = max(uFeather, 0.002);
    float c = clamp(0.5 + uThreshold, 0.05, 0.95);
    m = smoothstep(c - f, c + f, m);
    vec3 col = uUseMap ? texture2D(uMap, vUv).rgb : uColor;
    // El objetivo se pinta con el color o la imagen (intensidad 1 = lo reemplaza)
    // Sobre la cámara en vivo de 8th Wall: solo se pinta el objetivo
    gl_FragColor = vec4(col, m * uIntensity * uReady);
  }
`

// ---------- Estado ----------

type Estado = {
  mesh: any
  material: any
  maskTex: any
  maskData: Uint8Array
  camTex: any
  camData: Uint8Array
  mapTex: any
  imagenCargada: string
  motor: Motor | null
  ocupado: boolean
  ultimoFrameId: number
  prevMask: Float32Array | null
  objetivoPrevio: string
  avisoMostrado: boolean
  ms: number
  medidas: number
  activo: boolean
  flowTex: any
  flowData: Uint8Array
  grisActual: Gris | null     // frame actual en gris
  grisRef: Gris | null        // frame de la máscara que se muestra
  flujoWk: Worker
  flujoOcupado: boolean
  versionRef: number          // sube cada vez que llega una máscara nueva
  enEspera: {mascara: Float32Array, w: number, h: number} | null  // máscara que espera su flujo
  movimiento: number          // movimiento medio entre el frame actual y el analizado
  frameVivoId: number
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

// Sube la máscara con un poco de suavizado en el tiempo (sin parpadeo del borde)
const subirMascara = (st: Estado, u: any, mascara: Float32Array, w: number, h: number, quieto: boolean) => {
  if (st.maskTex.image.width !== w || st.maskTex.image.height !== h) {
    st.maskTex.dispose()
    const nuevo = crearTextura(w, h)
    st.maskTex = nuevo.tex
    st.maskData = nuevo.data
    u.uMask.value = nuevo.tex
    u.uMaskTexel.value.set(1 / w, 1 / h)
    st.prevMask = null
  }
  // Solo si la escena está quieta (si hay movimiento, mezclar dejaría una estela)
  const prev = quieto && st.prevMask && st.prevMask.length === mascara.length ? st.prevMask : null
  const out = st.maskData
  for (let p = 0; p < mascara.length; p++) {
    let v = mascara[p]
    v = v < 0 ? 0 : v > 1 ? 1 : v
    if (prev) v = v * 0.8 + prev[p] * 0.2
    mascara[p] = v
    const b = (v * 255) | 0
    const o = p * 4
    out[o] = b
    out[o + 1] = b
    out[o + 2] = b
    out[o + 3] = 255
  }
  st.prevMask = mascara
  st.maskTex.needsUpdate = true
}

// Sube el frame actual y ajusta el encuadre "cover" a la pantalla
const subirFrame = (st: Estado, u: any, pixeles: Uint8ClampedArray, w: number, h: number, aspPantalla: number) => {
  if (st.camTex.image.width !== w || st.camTex.image.height !== h) {
    st.camTex.dispose()
    const nueva = crearTextura(w, h)
    st.camTex = nueva.tex
    st.camData = nueva.data
    u.uCam.value = nueva.tex
  }
  st.camData.set(pixeles)
  st.camTex.needsUpdate = true
  const aspFrame = w / h
  if (aspFrame > aspPantalla) {
    const sx = aspPantalla / aspFrame
    u.uScale.value.set(sx, 1)
    u.uOffset.value.set((1 - sx) / 2, 0)
  } else {
    const sy = aspFrame / aspPantalla
    u.uScale.value.set(1, sy)
    u.uOffset.value.set(0, (1 - sy) / 2)
  }
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

    // @label Ajuste del borde (− expande / + contrae)
    // @min -0.3
    // @max 0.3
    ajusteBorde: ecs.f32,

    // @label Invertir
    invertir: ecs.boolean,
  },

  schemaDefaults: {
    objetivo: 'Fondo',
    usarImagen: false,
    colorR: 0.1,
    colorG: 0.0,
    colorB: 0.2,
    intensidad: 1,
    suavizado: 0.04,
    ajusteBorde: 0,
    invertir: false,
  },

  add: (world, component) => {
    const {data: maskData, tex: maskTex} = crearTextura(1, 1)
    const {data: camData, tex: camTex} = crearTextura(1, 1)
    const {data: flowData, tex: flowTex} = crearTextura(1, 1)
    flowData.set([128, 128, 0, 255])   // sin movimiento
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
        uFeather: {value: 0.04},
        uInvert: {value: false},
        uScale: {value: new THREE.Vector2(1, 1)},
        uOffset: {value: new THREE.Vector2(0, 0)},
        uMaskTexel: {value: new THREE.Vector2(1 / 256, 1 / 256)},
        uIntensity: {value: 1},
        uThreshold: {value: 0},
        uReady: {value: 0},
        uFlow: {value: flowTex},
      },
    })
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material)
    mesh.frustumCulled = false
    mesh.renderOrder = -1000  // detrás de tus objetos 3D
    world.three.scene.add(mesh)

    const st: Estado = {
      mesh,
      material,
      maskTex,
      maskData,
      camTex,
      camData,
      mapTex: null,
      imagenCargada: '',
      motor: null,
      ocupado: false,
      ultimoFrameId: -1,
      prevMask: null,
      objetivoPrevio: '',
      avisoMostrado: false,
      ms: 0,
      medidas: 0,
      activo: true,
      flowTex,
      flowData,
      grisActual: null,
      grisRef: null,
      flujoWk: new Worker(URL.createObjectURL(new Blob([CODIGO_FLUJO], {type: 'text/javascript'}))),
      flujoOcupado: false,
      versionRef: 0,
      enEspera: null,
      movimiento: 0,
      frameVivoId: -1,
      pos: new THREE.Vector3(),
      quat: new THREE.Quaternion(),
    }
    instancias.set(component.eid, st)
    const u0 = material.uniforms
    st.flujoWk.onmessage = (e: MessageEvent) => {
      st.flujoOcupado = false
      const r = e.data
      if (!st.activo || r.version !== st.versionRef) return   // flujo de una máscara anterior
      if (st.enEspera) {
        const {mascara, w, h} = st.enEspera
        st.enEspera = null
        subirMascara(st, u0, mascara, w, h, st.movimiento < 0.5)
        if (u0.uReady.value === 0) console.log('[segmentacion] Primera máscara', w, 'x', h)
        u0.uReady.value = 1
      }
      st.movimiento = r.movimiento
      subirFlujo(st, u0, r.flujo, r.w, r.h)
    }
    console.log('[segmentacion] Versión', VERSION)
    elegirMotor().then((m) => { if (st.activo) st.motor = m }).catch(() => {})
  },

  tick: (world, component) => {
    const st = instancias.get(component.eid)
    if (!st) return
    const s = component.schema
    const cam = world.three.activeCamera
    if (!cam) return
    const u = st.material.uniforms

    // 1) Plano a pantalla completa, detrás de tus objetos 3D
    const d = Math.min(cam.far * 0.9, 1000)
    const fov = cam.isPerspectiveCamera ? cam.fov : 60
    const alto = 2 * d * Math.tan(THREE.MathUtils.degToRad(fov / 2)) * 1.02
    cam.getWorldPosition(st.pos)
    cam.getWorldQuaternion(st.quat)
    st.mesh.position.copy(st.pos)
    st.mesh.quaternion.copy(st.quat)
    st.mesh.translateZ(-d)
    st.mesh.scale.set(alto * (cam.aspect || 1), alto, 1)

    // 2) Parámetros del panel
    const objetivo = OBJETIVOS[s.objetivo] || OBJETIVOS.Fondo
    const [r, g, b] = normalizarColor(s.colorR, s.colorG, s.colorB)
    u.uColor.value.setRGB(r, g, b)
    u.uFeather.value = s.suavizado
    u.uInvert.value = s.invertir
    u.uIntensity.value = s.intensidad
    // "Ajuste del borde": + ajusta más el objetivo pintado
    u.uThreshold.value = s.ajusteBorde
    if (s.objetivo !== st.objetivoPrevio) {
      st.objetivoPrevio = s.objetivo
      st.prevMask = null   // otro objetivo: no mezclar con la máscara anterior
    }
    if (s.usarImagen && s.imagen && s.imagen !== st.imagenCargada) {
      st.imagenCargada = s.imagen
      new THREE.TextureLoader().load(s.imagen, (tex: any) => {
        if (st.mapTex) st.mapTex.dispose()
        st.mapTex = tex
        u.uMap.value = tex
      })
    }
    u.uUseMap.value = s.usarImagen && !!st.mapTex

    // 3) Cámara de 8th Wall
    if (!conectarXR8()) {
      if (!st.avisoMostrado) {
        console.warn('[segmentacion] Esperando a XR8 (¿la cámara tiene AR activado?)')
        st.avisoMostrado = true
      }
      return
    }
    const img = frameCompartido.img
    if (!img) return
    const motor = st.motor

    // 4) Cada frame nuevo: colores actuales para el borde y movimiento desde el frame analizado
    if (frameCompartido.id !== st.frameVivoId) {
      st.frameVivoId = frameCompartido.id
      subirFrame(st, u, img.data, img.width, img.height, cam.aspect || 1)
      st.grisActual = aGris(img, null)
      if (st.grisRef && !st.flujoOcupado) {
        st.flujoOcupado = true
        st.flujoWk.postMessage({act: st.grisActual})
      }
    }

    if (!motor || !st.grisActual || st.ocupado || frameCompartido.id === st.ultimoFrameId) return

    // 5) Analizar el frame más reciente. Su máscara se muestra recién cuando ya se sabe cuánto
    //    se movió todo desde ese frame (máscara y flujo llegan juntos a la pantalla).
    st.ultimoFrameId = frameCompartido.id
    st.ocupado = true
    const grisEnvio = st.grisActual
    const t0 = performance.now()
    motor.analizar(img, objetivo)
      .then(({mascara, w: mw, h: mh}) => {
        if (!st.activo) return
        st.enEspera = {mascara, w: mw, h: mh}
        st.grisRef = grisEnvio
        st.versionRef++
        st.flujoOcupado = true
        st.flujoWk.postMessage({ref: grisEnvio, version: st.versionRef, act: st.grisActual})
        st.ms += performance.now() - t0
        st.medidas++
        if (st.medidas === 60) {
          console.log('[segmentacion] Tiempo medio por análisis:', (st.ms / 60).toFixed(1), 'ms —', motor.nombre)
          st.ms = 0
          st.medidas = 0
        }
      })
      .catch(e => console.warn('[segmentacion] Error analizando frame', e))
      .finally(() => { st.ocupado = false })
  },

  remove: (world, component) => {
    const st = instancias.get(component.eid)
    if (!st) return
    st.activo = false
    world.three.scene.remove(st.mesh)
    st.mesh.geometry.dispose()
    st.material.dispose()
    st.maskTex.dispose()
    st.camTex.dispose()
    st.flowTex.dispose()
    st.flujoWk.terminate()
    if (st.mapTex) st.mapTex.dispose()
    instancias.delete(component.eid)
    if (instancias.size === 0) desconectarXR8()
  },
})