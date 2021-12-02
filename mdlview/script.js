
const canvas = document.querySelector("#cnv")
let offscreen;
try {
	offscreen = canvas.transferControlToOffscreen()
} catch {}

const IsWorkerMode = !!offscreen

let worker = globalThis

if ( IsWorkerMode )
	worker = new Worker("./worker.js")

globalThis.__canvas = canvas
worker.postMessage({
	action: "canvas",
	data  : IsWorkerMode ? offscreen : '__canvas'
}, IsWorkerMode ? [offscreen] : undefined)

worker.addEventListener('message', ({data: {action, data}}) => {
	switch(action) {
		case 'loadTexture':
			const img = new Image()
			img.src = data.url
			img.onload = () => {
				const canvas = document.createElement('canvas')
				canvas.width = img.width
				canvas.height = img.height
				const ctx = canvas.getContext('2d')
				ctx.drawImage(img, 0, 0)
				const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height)
				worker.postMessage({ action: 'loadTextureResult', data: { rpcID: data.rpcID, imgData } })
			}
			
			break;
		case 'fps':
			$showFPS.textContent = data.toFixed(2)
			break
		
		case 'informNumObjects':
			$elNumObjectsShow.textContent = data
			break
	}
})

const $showFPS = document.querySelector("#showFPS")

let numObjects, numObjectsNext = 10
const $elNumObjects = document.querySelector('#numObjects')
const $elNumObjectsShow = document.querySelector('#numObjectsShow')
;(function numObjectsViewUpdate() {
	setTimeout(numObjectsViewUpdate, 100)
	worker.postMessage({action: 'setNumObjects', data: numObjectsNext})
})()
$elNumObjects.value = numObjectsNext
$elNumObjects.addEventListener('input', () => {
	const min = 10, max = 10000
		
	let value = +$elNumObjects.value
	if ( isNaN(value) )
		return
	numObjectsNext = Math.min(max, Math.max(min, value))
})


function watchInput() {
	const MOVE_FORWARD  = 0b1
	const MOVE_BACKWARD = 0b10
	const MOVE_LEFT     = 0b100
	const MOVE_RIGHT    = 0b1000
	const ROTATE_LEFT   = 0b10000
	const ROTATE_RIGHT  = 0b100000

	const keyMap = {
		38: MOVE_FORWARD, 87: MOVE_FORWARD,
		40: MOVE_BACKWARD, 83: MOVE_BACKWARD,
		81: MOVE_LEFT,
		69: MOVE_RIGHT,
		
		37: ROTATE_LEFT, 65: ROTATE_LEFT,
		39: ROTATE_RIGHT, 68: ROTATE_RIGHT,
	}
	
	let keyboardState = 0
	const setState = newInputState => {
		if ( newInputState === keyboardState ) return
		keyboardState = newInputState
		worker.postMessage({ action: 'keyboardState', data: keyboardState })
	}
	window.addEventListener('keydown', (e, bits = keyMap[ e.which ]) => setState( keyboardState | bits ) )
	window.addEventListener('keyup'  , (e, bits = keyMap[ e.which ]) => setState( (keyboardState | bits) ^ bits ) )
	
	let mousePressed = false
	window.addEventListener('mousemove', ({movementX: x, movementY: y}) => {
		if ( !mousePressed ) return
		worker.postMessage({ action: 'mousemove', data: {x, y} })
	})
	window.addEventListener('mousedown', e => mousePressed = true)
	window.addEventListener('mouseup', e => mousePressed = false)
	
	window.addEventListener('blur', e => (setState( 0 ), mousePressed = false) )
}
watchInput()

function watchAniList() {
	const $aniBtnList = [...document.querySelectorAll('.aniList > button')]
	const click = i => {
		const $el = $aniBtnList[i]
		$aniBtnList.map(v => v.classList.remove('active'))
		$el.classList.add('active')
		worker.postMessage({ action: 'setAniPattern', data: $el.textContent.trim().toLowerCase() })
	}
	$aniBtnList.map(($el, i) => $el.addEventListener('click', () => click(i)))
	click(3)
}
watchAniList()

function watchShowMode() {
	const $fullScreen = document.querySelector('.fullScreen button')
	$fullScreen.addEventListener('click', () => {
		$fullScreen.classList.toggle('active')
		const isFullScreen = $fullScreen.classList.contains('active')
			
		document.body.classList.toggle('modeFullScreen', isFullScreen)
	})
	;(function watchResize() {
		setTimeout(watchResize, 400)
		
		const { width, height } = canvas.getBoundingClientRect()
		worker.postMessage({ action: 'resize', data: { width, height } })
	})()
}
watchShowMode()


function parseLocationHash() {
	let obj = {}
	try {
		obj = Object.fromEntries( [...new URLSearchParams(location.hash.slice(1))] )
	} catch {}
	worker.postMessage({ action: 'setSettings', data: obj })
}
parseLocationHash()
globalThis.addEventListener('hashchange', parseLocationHash)
