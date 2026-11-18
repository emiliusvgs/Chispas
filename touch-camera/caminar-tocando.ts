import * as ecs from '@8thwall/ecs'

// Caminar tocando: mantén el dedo en la pantalla y avanzas hacia donde apunta el celular.
// Ponlo en el objeto padre "Calle" (no en el splat) y elige tu Camera en el campo "camera".
// Mirar alrededor y caminar de verdad siguen funcionando igual.

type State = {
  baseX: number
  baseY: number
  baseZ: number
  offX: number
  offZ: number
  pressing: boolean
  last: number
  cleanup: () => void
}

const states = new Map<number, State>()

ecs.registerComponent({
  name: 'caminar-tocando',
  schema: {
    camera: ecs.eid,
    speed: ecs.f32, // metros por segundo
  },
  schemaDefaults: {
    speed: 1.4, // paso de caminata normal
  },
  add: (world, component) => {
    const {eid} = component
    const p = ecs.Position.get(world, eid)
    const s: State = {
      baseX: p.x,
      baseY: p.y,
      baseZ: p.z,
      offX: 0,
      offZ: 0,
      pressing: false,
      last: performance.now(),
      cleanup: () => {},
    }
    const down = () => { s.pressing = true }
    const up = () => { s.pressing = false }
    window.addEventListener('touchstart', down, {passive: true})
    window.addEventListener('touchend', up, {passive: true})
    window.addEventListener('touchcancel', up, {passive: true})
    window.addEventListener('mousedown', down)
    window.addEventListener('mouseup', up)
    s.cleanup = () => {
      window.removeEventListener('touchstart', down)
      window.removeEventListener('touchend', up)
      window.removeEventListener('touchcancel', up)
      window.removeEventListener('mousedown', down)
      window.removeEventListener('mouseup', up)
    }
    states.set(eid, s)
  },
  tick: (world, component) => {
    const s = states.get(component.eid)
    if (!s) return
    const now = performance.now()
    const dt = Math.min((now - s.last) / 1000, 0.1)
    s.last = now

    if (s.pressing) {
      const camEid = component.schema.camera || (world as any).camera?.getActiveEid?.()
      if (camEid) {
        const q = ecs.Quaternion.get(world, camEid)
        // Dirección a la que mira la cámara (eje -Z), solo en el plano del suelo.
        let fx = -2 * (q.x * q.z + q.w * q.y)
        let fz = -(1 - 2 * (q.x * q.x + q.y * q.y))
        const len = Math.hypot(fx, fz)
        if (len > 0.001) {
          fx /= len
          fz /= len
          // La calle se mueve en sentido contrario = tú avanzas.
          s.offX -= fx * component.schema.speed * dt
          s.offZ -= fz * component.schema.speed * dt
        }
      }
    }

    ecs.Position.set(world, component.eid, {
      x: s.baseX + s.offX,
      y: s.baseY,
      z: s.baseZ + s.offZ,
    })
  },
  remove: (world, component) => {
    const s = states.get(component.eid)
    s?.cleanup()
    states.delete(component.eid)
  },
})
