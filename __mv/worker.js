try {
	importScripts('./mat4.js')
} catch {}


let forceNumMonsters = 0
let createDistMonsters = 100
let zDist = -100
let gAniSpeed = 10
let aniPattern = 'idle'


globalThis.gl = null;
globalThis.programInfo=null;
let glTextureFloat, glTextureFloatVertices;
function createFloatTexture() {
    var glTextureFloat = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, glTextureFloat);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 2048, 2048, 0, gl.RGBA, gl.FLOAT, null);
	return glTextureFloat
}
function main(canvas) {
	canvas.width = 640
	canvas.height = 480
	
	globalThis.canvas=canvas
	gl = canvas.getContext('webgl', {
		alpha: false,
		depth: true,
		stencil: false,
		antialias: false,
		premultipliedAlpha: true,
		preserveDrawingBuffer: true,
		//failIfMajorPerformanceCaveat: true,
	});

	if ( !gl ) {
		alert('Unable to initialize WebGL. Your browser or machine may not support it.');
		return
	}

  const vsSource = `
	attribute float aMatrixBase;

    attribute vec3 aVertexPosition;
	attribute vec3 aNormal;
	attribute vec2 aUV;
	attribute vec4 aBoneIndex;
	attribute vec4 aBoneWeight;
	
	attribute float aMatrixIndex;
	
	attribute float aVertexIndex;
    //attribute vec4 aVertexColor;

	uniform float uMatrixBase;
	
	uniform mat4 uModelViewMatrix;
    uniform mat4 uProjectionMatrix;
    
	uniform sampler2D uTextureFloat;
	uniform sampler2D uTextureFloatTest;
	uniform sampler2D uTextureFloatVertex;
	
	
	uniform vec2      uTextureFloatMatrixSIZE;
	uniform sampler2D uTextureFloatMatrix;
	
	uniform sampler2D uTextureDiff;

    varying highp vec3 vNormal;
    varying lowp vec4 vColor;
    varying highp vec2 vUV;
	
	#define FLOAT_TEXTURE_WH (512.0)

	highp mat4 tex_LoadMat4(sampler2D tex, highp float index, highp vec2 texSize) {
		index *= 4.0;

		highp float x = mod(index, texSize.x);
		highp float y = index / texSize.x;
		
		highp mat4 ret;
		ret[0] = texture2D(tex, vec2(x + 0.0, y) / texSize);
		ret[1] = texture2D(tex, vec2(x + 1.0, y) / texSize);
		ret[2] = texture2D(tex, vec2(x + 2.0, y) / texSize);
		ret[3] = texture2D(tex, vec2(x + 3.0, y) / texSize);
		
		return ret;
	}
	

    void main(void) {
		vec3 position; 
		vec3 normal = vec3(1, 0, 0);
		vec2 uv = vec2(0, 0);

		
		position = aVertexPosition;
		normal = aNormal;
		uv = aUV;

		vec4 pos = vec4(position, 1.0);

		mat4 worldMatrix = tex_LoadMat4(uTextureFloatMatrix, aMatrixBase + uMatrixBase + aMatrixIndex, uTextureFloatMatrixSIZE);
		//mat4 worldMatrix = tex_LoadMat4(uTextureFloatMatrix, aMatrixBase + uMatrixBase + aMatrixIndex, vec2(512.0, 256.0));
		
		//worldMatrix = worldMatrix * uModelViewMatrix;
		
		gl_Position = worldMatrix * pos;
		gl_Position = uProjectionMatrix * gl_Position;

		vNormal = mat3(worldMatrix) * normal;
		
		vUV = uv;
		vNormal.x += aMatrixBase*0.0;
    }
  `;

  // Fragment shader program

  const fsSource = `precision highp float;
	uniform sampler2D uTextureDiff;
	uniform sampler2D uTextureFloatTest;
    
	varying highp vec3 vNormal;
    varying lowp vec4 vColor;
    varying highp vec2 vUV;

    void main(void) {
		highp vec2 uv = vUV;
		//uv.x = 1.0 - uv.x;
		uv.y = 1.0 - uv.y;
		gl_FragColor = texture2D(uTextureDiff, uv);
	
		vec3 normal = normalize(vNormal);
		vec3 uReverseLightDirection = vec3(0,1,0);
		uReverseLightDirection = vec3(0, 0, 1);
		float light = dot(normal, uReverseLightDirection);
		
		gl_FragColor.rbg *= (0.7 + light * 0.3);
		//gl_FragColor.rbg *= light;
		
		//gl_FragColor[3]=1.0;
		if ( gl_FragColor[3] < 0.9 ) discard;
    }
  `;

  // Initialize a shader program; this is where all the lighting
  // for the vertices and so forth is established.
  const shaderProgram = initShaderProgram(gl, vsSource, fsSource);

  // Collect all the info needed to use the shader program.
  // Look up which attributes our shader program is using
  // for aVertexPosition, aVevrtexColor and also
  // look up uniform locations.
   programInfo = {
    program: shaderProgram,
    attribLocations: {
      aVertexPosition: gl.getAttribLocation(shaderProgram, 'aVertexPosition'),
      aNormal: gl.getAttribLocation(shaderProgram, 'aNormal'),
      aUV: gl.getAttribLocation(shaderProgram, 'aUV'),
      aBoneIndex: gl.getAttribLocation(shaderProgram, 'aBoneIndex'),
      aBoneWeight: gl.getAttribLocation(shaderProgram, 'aBoneWeight'),
      vertexColor: gl.getAttribLocation(shaderProgram, 'aVertexColor'),
      aVertexIndex: gl.getAttribLocation(shaderProgram, 'aVertexIndex'),
      aMatrixIndex: gl.getAttribLocation(shaderProgram, 'aMatrixIndex'),
      aMatrixBase: gl.getAttribLocation(shaderProgram, 'aMatrixBase'),
	    
    },
    uniformLocations: {
		projectionMatrix: gl.getUniformLocation(shaderProgram, 'uProjectionMatrix'),
		modelViewMatrix: gl.getUniformLocation(shaderProgram, 'uModelViewMatrix'),
		uProjectionMatrix: gl.getUniformLocation(shaderProgram, 'uProjectionMatrix'),
		uTextureFloat: gl.getUniformLocation(shaderProgram, 'uTextureFloat'),
		uTextureFloatVertices: gl.getUniformLocation(shaderProgram, 'uTextureFloatVertices'),
		uTextureDiff: gl.getUniformLocation(shaderProgram, 'uTextureDiff'),
		uTextureFloatTest: gl.getUniformLocation(shaderProgram, 'uTextureFloatTest'),
		uTextureFloatVertex: gl.getUniformLocation(shaderProgram, 'uTextureFloatVertex'),
		uTextureFloatMatrix: gl.getUniformLocation(shaderProgram, 'uTextureFloatMatrix'),
		uMatrixBase: gl.getUniformLocation(shaderProgram, 'uMatrixBase'),
		uTextureFloatMatrixSIZE: gl.getUniformLocation(shaderProgram, 'uTextureFloatMatrixSIZE'),
		  
		 
    },
  };


  const buffers = initBuffers(gl);
  globalThis.draw = () =>   drawScene(gl, programInfo, buffers)
}
function initBuffers(gl) {

	const positionBuffer = gl.createBuffer();


	gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

  // Now create an array of positions for the square.

  const positions = [
     1.0,  1.0, 0,
    -1.0,  1.0, 0,
     1.0, -1.0, 0,
    -1.0, -1.0, 0,
  ];

  // Now pass the list of positions into WebGL to build the
  // shape. We do this by creating a Float32Array from the
  // JavaScript array, then use it to fill the current buffer.

  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

  // Now set up the colors for the vertices

  var colors = [
    1.0,  1.0,  1.0,  1.0,    // white
    1.0,  0.0,  0.0,  1.0,    // red
    0.0,  1.0,  0.0,  1.0,    // green
    0.0,  0.0,  1.0,  1.0,    // blue
  ];

  const colorBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);

  return {
    position: positionBuffer,
    color: colorBuffer,
  };
}

let loadTexturesMap = {}, ltRPCID = 1
function glLoadTexture(url) {
	const glTex = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, glTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	
	let rpcID = ltRPCID++
	loadTexturesMap[rpcID] = imgData => {
		const level = 0;
		const internalFormat = gl.RGBA;
		const width = 1;
		const height = 1;
		const border = 0;
		const format = gl.RGBA;
		const type = gl.UNSIGNED_BYTE;
		gl.bindTexture(gl.TEXTURE_2D, glTex);
		gl.texImage2D(gl.TEXTURE_2D, level, internalFormat, format, type, imgData)
		gl.generateMipmap(gl.TEXTURE_2D)
		
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		
		if ( glExtTextureFilterAnisotropic )
			gl.texParameterf(gl.TEXTURE_2D, glExtTextureFilterAnisotropic.TEXTURE_MAX_ANISOTROPY_EXT, glExtTextureFilterAnisotropic.maxAnisotropy)
	}
	postMessage({ action: 'loadTexture', data: {url, rpcID} })
	return glTex
}

function nextPowOfTwo(num) {
	let n = num
	for(let _n = n; _n; n |= _n, _n >>>= 1);
	const ret = n + 1
	return ret === num << 1 ? num : ret
}
class FloatTexture {
	constructor(size = 1024, width = 1024) {
		this.width = width
		this.height = 2
		this.float32array = new Float32Array(0)
		this.glTextureFloat = null
		this.grow(size)
		this._create()
	}

	_create() {
		if ( this.glTextureFloat )
			gl.deleteTexture( this.glTextureFloat )
		
		this.glTextureFloat = gl.createTexture()
		gl.bindTexture(gl.TEXTURE_2D, this.glTextureFloat)
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.width, this.height, 0, gl.RGBA, gl.FLOAT, null)
	}
	
	writeVec3(index, vec3) {
		this.float32array[ index*4 + 0 ] = vec3[0]
		this.float32array[ index*4 + 1 ] = vec3[1]
		this.float32array[ index*4 + 2 ] = vec3[2]
	}
	
	writeFloat32(index, value) {
		this.float32array[index] = value
	}

	writeMatrix4x4(index) {
		this.float32array
	}

	write() {
	}

	swap() {
		gl.bindTexture(gl.TEXTURE_2D, this.glTextureFloat)
		gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.width, this.height, gl.RGBA, gl.FLOAT, this.float32array)
	}

	grow(newSize) {
		this.height = Math.max(4, nextPowOfTwo( Math.ceil(newSize / 4 / this.width) ) )
		const float32array = new Float32Array(this.width * this.height * 4)
		float32array.set( this.float32array )
		this.float32array = float32array
		this._create()
		console.log('FloatTexture: %s -> %sx%s len=%s', newSize, this.width, this.height, this.float32array.length)
	}
}

let mat4Decompose;
(() => {
	function normalize(out, mat) {
		var m44 = mat[15]
		// Cannot normalize.
		if (m44 === 0) 
			return false
		var scale = 1 / m44
		for (var i=0; i<16; i++)
			out[i] = mat[i] * scale
		return true
	}

	const { create, clone, determinant, invert, transpose } = mat4

	var tmp = create()
	var perspectiveMatrix = create()
	var tmpVec4 = [0, 0, 0, 0]
	var row = [ [0,0,0], [0,0,0], [0,0,0] ]
	var pdum3 = [0,0,0]

	mat4Decompose = function decomposeMat4(matrix, translation, scale, skew, perspective, quaternion) {
		if (!translation) translation = [0,0,0]
		if (!scale) scale = [0,0,0]
		if (!skew) skew = [0,0,0]
		if (!perspective) perspective = [0,0,0,1]
		if (!quaternion) quaternion = [0,0,0,1]

		//normalize, if not possible then bail out early
		if (!normalize(tmp, matrix))
			return false

		// perspectiveMatrix is used to solve for perspective, but it also provides
		// an easy way to test for singularity of the upper 3x3 component.
		clone(perspectiveMatrix, tmp)

		perspectiveMatrix[3] = 0
		perspectiveMatrix[7] = 0
		perspectiveMatrix[11] = 0
		perspectiveMatrix[15] = 1

		// If the perspectiveMatrix is not invertible, we are also unable to
		// decompose, so we'll bail early. Constant taken from SkMatrix44::invert.
		if (Math.abs(determinant(perspectiveMatrix) < 1e-8))
			return false

		var a03 = tmp[3], a13 = tmp[7], a23 = tmp[11],
				a30 = tmp[12], a31 = tmp[13], a32 = tmp[14], a33 = tmp[15]

		// First, isolate perspective.
		if (a03 !== 0 || a13 !== 0 || a23 !== 0) {
			tmpVec4[0] = a03
			tmpVec4[1] = a13
			tmpVec4[2] = a23
			tmpVec4[3] = a33

			// Solve the equation by inverting perspectiveMatrix and multiplying
			// rightHandSide by the inverse.
			// resuing the perspectiveMatrix here since it's no longer needed
			var ret = invert(perspectiveMatrix, perspectiveMatrix)
			if (!ret) return false
			transpose(perspectiveMatrix, perspectiveMatrix)

			//multiply by transposed inverse perspective matrix, into perspective vec4
			vec4multMat4(perspective, tmpVec4, perspectiveMatrix)
		} else { 
			//no perspective
			perspective[0] = perspective[1] = perspective[2] = 0
			perspective[3] = 1
		}

		// Next take care of translation
		translation[0] = a30
		translation[1] = a31
		translation[2] = a32

		// Now get scale and shear. 'row' is a 3 element array of 3 component vectors
		mat3from4(row, tmp)

		// Compute X scale factor and normalize first row.
		scale[0] = vec3.length(row[0])
		vec3.normalize(row[0], row[0])

		// Compute XY shear factor and make 2nd row orthogonal to 1st.
		skew[0] = vec3.dot(row[0], row[1])
		combine(row[1], row[1], row[0], 1.0, -skew[0])

		// Now, compute Y scale and normalize 2nd row.
		scale[1] = vec3.length(row[1])
		vec3.normalize(row[1], row[1])
		skew[0] /= scale[1]

		// Compute XZ and YZ shears, orthogonalize 3rd row
		skew[1] = vec3.dot(row[0], row[2])
		combine(row[2], row[2], row[0], 1.0, -skew[1])
		skew[2] = vec3.dot(row[1], row[2])
		combine(row[2], row[2], row[1], 1.0, -skew[2])

		// Next, get Z scale and normalize 3rd row.
		scale[2] = vec3.length(row[2])
		vec3.normalize(row[2], row[2])
		skew[1] /= scale[2]
		skew[2] /= scale[2]


		// At this point, the matrix (in rows) is orthonormal.
		// Check for a coordinate system flip.  If the determinant
		// is -1, then negate the matrix and the scaling factors.
		vec3.cross(pdum3, row[1], row[2])
		if (vec3.dot(row[0], pdum3) < 0) {
			for (var i = 0; i < 3; i++) {
				scale[i] *= -1;
				row[i][0] *= -1
				row[i][1] *= -1
				row[i][2] *= -1
			}
		}

		// Now, get the rotations out
		quaternion[0] = 0.5 * Math.sqrt(Math.max(1 + row[0][0] - row[1][1] - row[2][2], 0))
		quaternion[1] = 0.5 * Math.sqrt(Math.max(1 - row[0][0] + row[1][1] - row[2][2], 0))
		quaternion[2] = 0.5 * Math.sqrt(Math.max(1 - row[0][0] - row[1][1] + row[2][2], 0))
		quaternion[3] = 0.5 * Math.sqrt(Math.max(1 + row[0][0] + row[1][1] + row[2][2], 0))

		if (row[2][1] > row[1][2])
			quaternion[0] = -quaternion[0]
		if (row[0][2] > row[2][0])
			quaternion[1] = -quaternion[1]
		if (row[1][0] > row[0][1])
			quaternion[2] = -quaternion[2]
		return true
	}

	//will be replaced by gl-vec4 eventually
	function vec4multMat4(out, a, m) {
		var x = a[0], y = a[1], z = a[2], w = a[3];
		out[0] = m[0] * x + m[4] * y + m[8] * z + m[12] * w;
		out[1] = m[1] * x + m[5] * y + m[9] * z + m[13] * w;
		out[2] = m[2] * x + m[6] * y + m[10] * z + m[14] * w;
		out[3] = m[3] * x + m[7] * y + m[11] * z + m[15] * w;
		return out;
	}

	//gets upper-left of a 4x4 matrix into a 3x3 of vectors
	function mat3from4(out, mat4x4) {
		out[0][0] = mat4x4[0]
		out[0][1] = mat4x4[1]
		out[0][2] = mat4x4[2]
		
		out[1][0] = mat4x4[4]
		out[1][1] = mat4x4[5]
		out[1][2] = mat4x4[6]

		out[2][0] = mat4x4[8]
		out[2][1] = mat4x4[9]
		out[2][2] = mat4x4[10]
	}

	function combine(out, a, b, scale1, scale2) {
		out[0] = a[0] * scale1 + b[0] * scale2
		out[1] = a[1] * scale1 + b[1] * scale2
		out[2] = a[2] * scale1 + b[2] * scale2
	}
		
})()
const __mat4DecomposeSimpleEqEps = (a, b, eps) => Math.abs(a - b) < eps
const __mat4DecomposeSimpleSkew = [0,0,0], __mat4DecomposeSimplePerspective = [0,0,0,1]
const mat4DecomposeSimple = (mat4, quat, pos, scale) => {
	if ( !mat4Decompose(mat4, pos, scale, __mat4DecomposeSimpleSkew, __mat4DecomposeSimplePerspective, quat) )
		return false

	if ( !__mat4DecomposeSimpleSkew.every(n => __mat4DecomposeSimpleEqEps(n, 0, 0.001)) )
		return false

	const p = __mat4DecomposeSimplePerspective
	if ( !( __mat4DecomposeSimpleEqEps(p[0], 0, 0.0001) &&
			__mat4DecomposeSimpleEqEps(p[1], 0, 0.0001) &&
			__mat4DecomposeSimpleEqEps(p[2], 0, 0.0001) &&
			__mat4DecomposeSimpleEqEps(p[3], 1, 0.0001) ) )
		return false

	return true
}


function glSetTexture(location, index, glTexture) {
	gl.activeTexture(gl.TEXTURE0 + index)
	gl.bindTexture(gl.TEXTURE_2D, glTexture)
	gl.uniform1i(location, index);
}
function clear() {
}

class ViewPort {
	width  = 640
	height = 480
	canvas = null
	gl = null

	init(canvas, gl) {
		this.canvas = canvas
		this.gl = gl
		this.resize()
	}
	
	resize() {
		if ( this.canvas ) {
			this.canvas.width  = this.width
			this.canvas.height = this.height
		}
		
		if ( this.gl ) {
			this.gl.viewport(0, 0, this.width, this.height)
		}
	}
	
	update({width, height}) {
		if ( this.width === width && this.height === height )
			return

		this.width  = width
		this.height = height
		this.resize()
	}
}
const viewPort = new ViewPort()

class Camera {
	constructor() {
		this.pos = [0, 0, 0]
		this.rotateX = 0
		this.rotateZ = 0
	}
	
	mousemove = ({x, y}) => {
		this.rotateX += y
		this.rotateZ += x
	}
	
	update(obj) {
		['rotateX', 'rotateZ'].map(k => this[k] = +(obj[k] ?? this[k]))
	}
}
const camera = new Camera()

const modelViewGLFormat = mat4.create();
mat4.rotateX(modelViewGLFormat, modelViewGLFormat, Math.PI/180 * -90)

const modelViewMatrix = mat4.create()
const projectionMatrix = mat4.create()
function loopView() {
	globalThis.rX = globalThis.rX || 0
	globalThis.rY = globalThis.rY || 0
	
	const matRotateZ = mat4.create();
	const matRotateX = mat4.create();
	const matRotate = mat4.create();
	mat4.rotateZ(matRotateZ, matRotateZ, Math.PI/180 * (rY + camera.rotateZ))	
	mat4.rotateX(matRotateX, matRotateX, Math.PI/180 * (rX + camera.rotateX))
	
	mat4.multiply(matRotate, matRotateX, matRotateZ)
	mat4.multiply(matRotate, modelViewGLFormat, matRotate)
	
	
	const matTranslate = mat4.create();
	//mat4.translate(matTranslate, matTranslate, [0, -15, -40])
	//mat4.translate(matTranslate, matTranslate, [0, 0, -500])
	mat4.translate(matTranslate, matTranslate, [0, -10, -40])
	mat4.translate(matTranslate, matTranslate, [0, 0, +10])
	mat4.translate(matTranslate, matTranslate, [0, 0, zDist])
	
	mat4.identity(modelViewMatrix)
	mat4.multiply(modelViewMatrix, matRotate, modelViewMatrix)
	mat4.multiply(modelViewMatrix, matTranslate, modelViewMatrix)
	
	///// perspective
	const fieldOfView = 45 * Math.PI / 180;   // in radians
	const aspect = viewPort.width / viewPort.height
	const zNear = 0.1;
	const zFar = 1000.0;
	mat4.perspective(projectionMatrix, fieldOfView, aspect, zNear, zFar);	
}

function drawScene(gl, programInfo, buffers) {
	//gl.clearColor(0.0, 0.0, 0.0, 1.0)
	gl.clearDepth(1.0)
	gl.enable(gl.DEPTH_TEST)
	gl.depthFunc(gl.LEQUAL)
	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

	gl.enable( gl.BLEND );
	gl.blendEquation( gl.FUNC_ADD );
	gl.blendFunc( gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA );
	gl.blendFunc( gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA );
	//gl.disable( gl.BLEND )
	
	gl.enable(gl.CULL_FACE)
	gl.cullFace(gl.BACK)
	gl.disable(gl.CULL_FACE)
}
function draw2(i) {
	gl.useProgram(programInfo.program)
	gl.uniformMatrix4fv(programInfo.uniformLocations.uProjectionMatrix, false, projectionMatrix)
	gl.uniformMatrix4fv(programInfo.uniformLocations.uModelViewMatrix , false, mat4.create())
	
	glSetTexture(programInfo.uniformLocations.uTextureFloatMatrix, 5, glFloatTextureMatrix.floatTexture.glTextureFloat)

	gl.enableVertexAttribArray( programInfo.attribLocations.aVertexPosition )
	gl.enableVertexAttribArray( programInfo.attribLocations.aNormal )
	gl.enableVertexAttribArray( programInfo.attribLocations.aUV )
	gl.enableVertexAttribArray( programInfo.attribLocations.aMatrixIndex )
	gl.enableVertexAttribArray( programInfo.attribLocations.aMatrixBase )

	const type = gl.FLOAT;
	const normalize = false;
	const stride = (3 + 3 + 2 + 1)*4		
	gl.bindBuffer(gl.ARRAY_BUFFER, glBuffer.glBuffer)
	gl.vertexAttribPointer( programInfo.attribLocations.aVertexPosition, 3, type, normalize, stride, 0 )
	gl.vertexAttribPointer( programInfo.attribLocations.aNormal        , 3, type, normalize, stride, (3)*4 )
	gl.vertexAttribPointer( programInfo.attribLocations.aUV            , 2, type, normalize, stride, (3+3)*4 )
	gl.vertexAttribPointer( programInfo.attribLocations.aMatrixIndex   , 1, type, normalize, stride, (3+3+2)*4 )	
	
	gl.bindBuffer(gl.ARRAY_BUFFER, glBufferInstancedFloat.glBuffer)
	gl.vertexAttribPointer(programInfo.attribLocations.aMatrixBase, 1, gl.FLOAT, false, 0, 0)
	glExtInstancedArrays.vertexAttribDivisorANGLE(programInfo.attribLocations.aMatrixBase, 1)

	if( 0 )
	monsters.map(m => {
		gl.uniform1f(programInfo.uniformLocations.uMatrixBase, m.matrixStart);
		m.draw()
	})

	if ( 0 )
	monsters.map((m, i) => {
		gl.uniform1f(programInfo.uniformLocations.uMatrixBase, m.matrixStart);
		m.monster.groupByTexture.map(mesh => {
			glSetTexture(programInfo.uniformLocations.uTextureDiff, 1, mesh.glTexture)
			gl.drawArrays(gl.TRIANGLES, mesh.vertexStart, mesh.vertexCount)
		})
		
	})
	
	if( 1 ) {
		const groupByTexture = {}
		monsters.map(monster => {
			monster.monster.groupByTexture.map(group => {
				const newGroup = groupByTexture[group.textureName] = groupByTexture[group.textureName] || { meshs: [] }
				newGroup.glTexture   = group.glTexture
				newGroup.vertexStart = group.vertexStart
				newGroup.vertexCount = group.vertexCount
				newGroup.meshs.push({ matrixStart: monster.matrixStart })
			})
		})
		
		let j = 0
		for(let i in groupByTexture) {
			const group = groupByTexture[i]
			group.instancedStart = j
			for(const {matrixStart} of group.meshs)
				glBufferInstancedFloat.float32array[j++] = matrixStart
			group.instancedCount = j - group.instancedStart
		}
		glBufferInstancedFloat.setMaxIndex(j)
		glBufferInstancedFloat.swap()
		
		gl.uniform1f(programInfo.uniformLocations.uMatrixBase, 0)
		
		gl.bindBuffer(gl.ARRAY_BUFFER, glBufferInstancedFloat.glBuffer)
		gl.vertexAttribPointer(programInfo.attribLocations.aMatrixBase, 1, gl.FLOAT, false, 0, 0)
		glExtInstancedArrays.vertexAttribDivisorANGLE(programInfo.attribLocations.aMatrixBase, 1)		
		
		gl.bindBuffer(gl.ARRAY_BUFFER, glBufferInstancedFloat.glBuffer)
		for(let i in groupByTexture) {
			const group = groupByTexture[i]
			glSetTexture(programInfo.uniformLocations.uTextureDiff, 1, group.glTexture)

			gl.vertexAttribPointer(programInfo.attribLocations.aMatrixBase, 1, gl.FLOAT, false, 0, group.instancedStart*4)
			glExtInstancedArrays.drawArraysInstancedANGLE(gl.TRIANGLES, group.vertexStart, group.vertexCount, group.instancedCount)	
		}
	}
}

function initShaderProgram(gl, vsSource, fsSource) {
  const vertexShader = loadShader(gl, gl.VERTEX_SHADER, vsSource);
  const fragmentShader = loadShader(gl, gl.FRAGMENT_SHADER, fsSource);

  // Create the shader program

  const shaderProgram = gl.createProgram();
  gl.attachShader(shaderProgram, vertexShader);
  gl.attachShader(shaderProgram, fragmentShader);
  gl.linkProgram(shaderProgram);

  // If creating the shader program failed, alert

  if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
    console.error('Unable to initialize the shader program: ' + gl.getProgramInfoLog(shaderProgram));
    return null;
  }

  return shaderProgram;
}
function loadShader(gl, type, source) {
  const shader = gl.createShader(type);

  // Send the source to the shader object

  gl.shaderSource(shader, source);

  // Compile the shader program

  gl.compileShader(shader);

  // See if it compiled successfully

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error('An error occurred compiling the shaders: ' + gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

function Fps() {
	const times = []
	let prevTime = null
	
	setInterval(() => {
		const avg = times.reduce((s, v) => s + v, 0) / times.length
		postMessage({action: 'fps', data: (1/avg*1e3)})
		//console.log('Fps avg: ' + (1/avg*1e3).toFixed(2))
	}, 1e3)
	
	return () => {
		const now = Date.now()
		if ( prevTime !== null )
			times.push(now - prevTime)
		prevTime = now

		if ( times.length > 100 )
			times.splice(0, times.length - 60)
	}
}
const mat4GetRotation = (out, mat) => {
    var m11 = mat[0],
      m12 = mat[1],
      m13 = mat[2],
      m21 = mat[4],
      m22 = mat[5],
      m23 = mat[6],
      m31 = mat[8],
      m32 = mat[9],
      m33 = mat[10];
   var s1 =1/ Math.sqrt(m11 * m11 + m12 * m12 + m13 * m13);
   var s2= 1/Math.sqrt(m21 * m21 + m22 * m22 + m23 * m23);
   var s3 = 1/Math.sqrt(m31 * m31 + m32 * m32 + m33 * m33);
  var trace = mat[0]*s1 + mat[5]*s2 + mat[10]*s3;
  var S = 0;
  if (trace > 0) { 
    S = Math.sqrt(trace + 1.0) * 2;
    out[3] = 0.25 * S;
    out[0] = (mat[6]*s3 - mat[9]*s2) / S;
    out[1] = (mat[8]*s1 - mat[2]*s3) / S; 
    out[2] = (mat[1]*s2 - mat[4]*s1) / S; 
  } else if ((mat[0]*s1 > mat[5]*s2)&(mat[0] *s1> mat[10]*s3)) { 
    S = Math.sqrt(1.0 + mat[0]*s1 - mat[5]*s2- mat[10]*s3) * 2;
    out[3] = (mat[6]*s3 - mat[9]*s2) / S;
    out[0] = 0.25 * S;
    out[1] = (mat[1]*s2 + mat[4]*s1) / S; 
    out[2] = (mat[8]*s1 + mat[2]*s3) / S; 
  } else if (mat[5]*s2 > mat[10]*s3) { 
    S = Math.sqrt(1.0 + mat[5]*s2 - mat[0]*s1 - mat[10]*s3) * 2;
    out[3] = (mat[8]*s1 - mat[2]*s3) / S;
    out[0] = (mat[1]*s2 + mat[4]*s1) / S; 
    out[1] = 0.25 * S;
    out[2] = (mat[6]*s3 + mat[9]*s2) / S; 
  } else { 
    S = Math.sqrt(1.0 + mat[10]*s3 - mat[0] *s1- mat[5]*s2) * 2;
    out[3] = (mat[1]*s2 - mat[4]*s1) / S;
    out[0] = (mat[8]*s1 + mat[2]*s3) / S;
    out[1] = (mat[6]*s3 + mat[9]*s2) / S;
    out[2] = 0.25 * S;
  }

  return out;
};


async function loadObj(url) {
	const obj = await (await fetch(url)).text()
	//console.log(obj)
	
	const lines = obj
		.split(/[\r\n]+/)
		.map(l => l.trim())
		.filter(l => l.length)
		
	const vertices = lines
		.filter(l => /^v\b/.test(l))
		.map(l => l.split(/\s+/).slice(1).map(Number))
	
	const faces = lines
		.filter(l => /^f\b/.test(l))
		.map(l => l.split(/\s+/).slice(1)
			.map(c => c.split('/')[0])
			.map(Number) )
		
	const vertexList = faces
		.map(t => t.map(i => vertices[i-1]))
	
	const vertexBuffer = new Float32Array( vertexList.flat(1e9) )
	
	//console.log( vertexBuffer )
	
	return vertexBuffer
}

class GLBuffer {
	constructor() {
		this.float32array = new Float32Array(10*1024*1024)
		this.allocIndex = 0
		this.glBuffer = null
	}
	
	alloc(float32array) {
		if ( this.allocIndex + float32array.length > this.float32array.length )
			throw new Error(`GLBuffer overflow`)
		
		const offset = this.allocIndex
		this.float32array.set(float32array, this.allocIndex)
		this.allocIndex += float32array.length
		this.swap()
		
		return offset
	}
	
	swap() {
		if ( this.glBuffer )
			gl.deleteBuffer(this.glBuffer)

		this.glBuffer = gl.createBuffer()
		gl.bindBuffer(gl.ARRAY_BUFFER, this.glBuffer)
		gl.bufferData(gl.ARRAY_BUFFER, this.float32array.subarray(0, this.allocIndex), gl.STATIC_DRAW)
	}
}
const glBuffer = new GLBuffer()

class GLBufferInstancedFloat {
	constructor() {
		this.float32array = new Float32Array(1024*1024)
		this.allocIndex = 0
		this.glBuffer = null
		this.maxIndex = 0
	}
	init() {
		this.glBuffer = gl.createBuffer()
		gl.bindBuffer(gl.ARRAY_BUFFER, this.glBuffer)
		gl.bufferData(gl.ARRAY_BUFFER, this.float32array, gl.STATIC_DRAW)
	}
	
	set(index, value) {
		this.float32array[index] = value
	}
	
	setMaxIndex(index) {
		this.maxIndex = index
	}
	
	swap() {
		gl.bindBuffer(gl.ARRAY_BUFFER, this.glBuffer)
		gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.float32array.subarray(0, this.maxIndex))
	}
}
const glBufferInstancedFloat = new GLBufferInstancedFloat()


class GLFloatTextureMatrix {
	constructor() {
	}
	init() {
		this.floatTexture = new FloatTexture()
		this.allocIndex = 0
	}
	
	alloc(num) {
		const size = num*16
		if ( this.allocIndex + size > this.floatTexture.length ) 
			throw new Error('GLFloat texture overflow')
		
		const start = this.allocIndex / 16
		const matrices = []
		
		if ( this.allocIndex + num * 16 > this.floatTexture.float32array.length ) {
			this.floatTexture.grow( this.allocIndex + num * 16 )
		}
		for(let i = 0; i < num; i++)
			matrices[i] = this.floatTexture.float32array.subarray(this.allocIndex + i * 16, this.allocIndex + i * 16 + 16)
		this.allocIndex += size
		return { start, matrices }
	}
	getMemory(start) {
		return this.floatTexture.float32array.subarray(start * 16)
	}
}
const glFloatTextureMatrix = new GLFloatTextureMatrix()

class Skeleton {
	constructor(bnData) {
		this.bnData = bnData
		this.bones = bnData.map(bone => ({...bone}))
		this.bones.map(bone => {
			bone.parent = this.bones.find(b => b.name === bone.parentName)
		})
		
		const bones = []
		this.bones
			.filter(b => !b.parent)
			.map(bone => {
				bones.push(bone)
				this.bones.splice(this.bones.indexOf(bone), 1)
			})

		loop:
		while( this.bones.length ) {	
			for(const bone of this.bones) {
				if ( bones.find(b => b === bone.parent) ) {
					bones.push(bone)
					this.bones.splice(this.bones.indexOf(bone), 1)
					continue loop;
				}
			}
		}
		
		this.bones = bones
		this.bones.map(bone => bone.parentIndex = this.bones.indexOf(bone.parent))
		this.bones.map(bone => bone.boneName = bone.name)
		
		this.bones.map(bone => {
			bone.globalMat4x4 = mat4.create()
			bone.localMat4x4 = new Float32Array(objToArray(bone.localMat4x4))
		})
		this.isValid = this._test()
		
		this.calcGlobalMatrix()
		this.calcDecompose()
		this.calcMatrix()
		
		this.bones.map(bone => {
			bone.globalInvertMat4x4 = mat4.create()
			mat4.invert(bone.globalInvertMat4x4, bone.globalMat4x4)
			
			bone.localInvertMat4x4 = mat4.create()
			mat4.invert(bone.localInvertMat4x4, bone.localMat4x4)
		})
	}
	
	getBoneIndexByName(boneName) {
		return this.bones.findIndex(b => b.name === boneName)
	}
	
	calcGlobalMatrix() {
		for(const bone of this.bones) {
			if ( !bone.parent ) {
				mat4.copy(bone.globalMat4x4, bone.localMat4x4)
				continue
			}
			
			mat4.multiply(bone.globalMat4x4, bone.parent.globalMat4x4, bone.localMat4x4)
		}
	}
	_test() {
		for(const bone of this.bones) {
			const matrix = bone.localMat4x4
			
			const localPos = vec3.create()
			const localQuat = quat.create()
			const localScale = vec3.create()
			if ( !mat4DecomposeSimple(matrix, localQuat, localPos, localScale) )
				return false

			const matrix2 = mat4.create()
			mat4.fromRotationTranslationScale(matrix2, localQuat, localPos, localScale)
				
			const vec3_1 = [10, 10, 10]
			const vec3_2 = [10, 10, 10]
			vec3.transformMat4(vec3_1, vec3_1, matrix)
			vec3.transformMat4(vec3_2, vec3_2, matrix2)

			if ( vec3_1.some((v, i) => Math.abs(vec3_2[i] - v) > 0.01) ) {
				console.log(matrix)
				return false
			}
		}
		console.log('ok')
		return true
	}
	calcDecompose() {
		for(const bone of this.bones) {
			bone.localPos = new Float32Array(3)
			bone.localQuat = new Float32Array(4)
			bone.localScale = new Float32Array([1,1,1])
		
			mat4.getTranslation(bone.localPos, bone.localMat4x4)
			mat4.getRotation(bone.localQuat, bone.localMat4x4)
			mat4.getScaling(bone.localScale, bone.localMat4x4)
		
			if ( !mat4DecomposeSimple(bone.localMat4x4, bone.localQuat, bone.localPos, bone.localScale) ) {
			}
		}
		
	}
	calcMatrix() {
		for(const bone of this.bones) {
			mat4.identity( bone.localMat4x4 )
			mat4.fromRotationTranslationScale(bone.localMat4x4, bone.localQuat, bone.localPos, bone.localScale)
		}
	}
	
	
	loop() {
		for(const bone of this.bones) {
			if ( !bone.parent ) {
				mat4.copy(bone.globalMat4x4, bone.localMat4x4)
				continue
			}
			
			mat4.multiply(bone.globalMat4x4, bone.parent.globalMat4x4, bone.localMat4x4)
		}
	}
}
class SkeletonInstance {
	constructor(skeleton) {
		this.skeleton = skeleton
		this.bones = this.skeleton.bones.map(bone => {
			const localPos   = new Float32Array(3)
			const localQuat  = new Float32Array(4)
			const localScale = new Float32Array(3)

			const globalInvertMat4x4 = mat4.create()
			const localInvertMat4x4 = mat4.create()
			const localMat4x4        = mat4.create()
			const globalMat4x4       = mat4.create()
			const finalMat4x4        = mat4.create()
			
			const parentIndex = bone.parentIndex
			const parent = null

			mat4.copy(globalInvertMat4x4, bone.globalInvertMat4x4)
			mat4.copy(localInvertMat4x4, bone.localInvertMat4x4)
			
			return { localPos, localQuat, localScale, globalInvertMat4x4, localInvertMat4x4, localMat4x4, globalMat4x4, finalMat4x4,
				parentIndex, parent,
			}
		})
		
		this.bones.map(bone => bone.parent = this.bones[bone.parentIndex])
		
		this.identity()
	}
	
	identity() {
		const { bones } = this.skeleton
		for(let i = 0; i < this.bones.length; i++) {
			const dstBone = this.bones[i]
			const srcBone = bones[i]
			vec3.copy(dstBone.localPos, srcBone.localPos)
			quat.copy(dstBone.localQuat, srcBone.localQuat)
			vec3.copy(dstBone.localScale, srcBone.localScale)
		}
	}

	compose() {
		for(const bone of this.bones) {
			mat4.fromRotationTranslationScale(bone.localMat4x4, bone.localQuat, bone.localPos, bone.localScale)
			if ( !bone.parent ) {
				mat4.copy(bone.globalMat4x4, bone.localMat4x4)
			} else {
				mat4.multiply(bone.globalMat4x4, bone.parent.globalMat4x4, bone.localMat4x4)
			}

			mat4.multiply(bone.finalMat4x4, bone.globalMat4x4, bone.globalInvertMat4x4)
		}
	}
}
class Animation {
	constructor(skeleton, aniData) {
		this.skeleton = skeleton
		this.aniData = aniData
		
		const MIN_FRAMES_DELTA = 160
		
		const list = aniData.filter(a => !this.skeleton.bones.find(b => b.boneName === a.boneName))
		console.log(list)
		
		const aniBones = this.skeleton.bones.map(bone => {
			const ani = {
				framePosArray: [],
				frameQuatArray: [],
				frameScaleArray: [],
				frameScaleUnkArray: [],
			}
			const aniForBone = aniData.find(a => a.boneName === bone.boneName)
			if ( aniForBone ) {
				ani.framePosArray = aniForBone.framePosArray || []
				ani.frameQuatArray = aniForBone.frameQuatArray || []
				ani.frameScaleArray = aniForBone.frameUnkVec3Array || []
				ani.frameScaleUnkArray = aniForBone.frameScaleArray || []
			}
			return ani
		})
		
		const aniBoneToArr = fani => [fani.framePosArray, fani.frameQuatArray, fani.frameScaleArray, fani.frameScaleUnkArray]
		const deltaFrameList = []
		let maxFrame = 0
		aniBones.map(fani => {
			aniBoneToArr(fani).map(frs => {
				if ( !frs.length ) 
					return
				
				for(let i = 1; i < frs.length; i++) {
					const prev = frs[i-1]
					const next = frs[i]
					if ( prev.frame === next.frame ) {
						if ( JSON.stringify(prev) !== JSON.stringify(next) )
							throw new Error(`#1 Invalid animations...`)
						
						frs.splice(i, 1)
						i -= 2
					}
				}
				
				if ( frs.length < 2 )
					throw new Error(`#2 Invalid animations...`)
				
				if ( frs[0].frame !== 0 )
					throw new Error(`#3 Invalid animations...`)
				
				frs.first = frs.shift()
				
				let prev = 0
				for(let i = 0; i < frs.length; i++) {
					const next = frs[i].frame
					const delta = next - prev
					prev = next
					
					deltaFrameList.push(delta)
				}
				maxFrame = Math.max(maxFrame, frs[frs.length - 1].frame)
			})			
		})
		deltaFrameList.sort((l, r) => l - r)
		if ( Math.min(...deltaFrameList) !== MIN_FRAMES_DELTA )
			throw new Error(`#4 Invalid animations...`)
		
		if ( deltaFrameList.every(v => v % MIN_FRAMES_DELTA !== 0) )
			throw new Error(`#5 Invalid animations...`)
		
		const tmpPos = [0,0,0], tmpPos2 = [0,0,0], tmpPos3 = [0,0,0]
		const numFramesForBone = maxFrame / MIN_FRAMES_DELTA
		const aniBonesNormalize = []
		
		aniBones.map(fani => {
			const lerp = (array, cb) => {
					for(let i = 0; i < array.length - 1; i++) {
						const rawFrame = array[i]
						const nextRawFrame = array[i + 1] || array[0]

						const index = rawFrame.frame / MIN_FRAMES_DELTA - 1
						const nextIndex = nextRawFrame.frame / MIN_FRAMES_DELTA - 1
						for(let j = index; j <= nextIndex; j++) {
							const f = 1 - (nextIndex - j) / (nextIndex - index)
							cb(j, rawFrame, nextRawFrame, f)
						}
					}
					
					const aFrame = array[ array.length - 1 ]
					const bFrame = array[0]
					const minFrameIndex = bFrame.frame / MIN_FRAMES_DELTA - 1
					for(let i = 0; i < minFrameIndex; i ++) {
						cb(i, bFrame, bFrame, 0)
					}
					const maxFrameIndex = aFrame.frame / MIN_FRAMES_DELTA - 1
					for(let i = maxFrameIndex; i < numFramesForBone; i++) {
						cb(i, aFrame, aFrame, 0)
					}
				}
			
			let posFrames, quatFrames, scaleFrames
			{
				const array = fani.framePosArray
				if ( array.length ) {
					posFrames = new Float32Array( numFramesForBone * 3 )
					lerp(array, (i, aFrame, bFrame, f) => vec3.lerp(posFrames.subarray(i*3), aFrame.pos, bFrame.pos, f))
				}
			}
			{
				const array = fani.frameQuatArray
				if ( array.length ) {
					fani.frameQuatArray.map(a => {
						quat.normalize(a.quat, a.quat)
						quat.invert(a.quat, a.quat)
						quat.normalize(a.quat, a.quat)
					})

					quatFrames = new Float32Array( numFramesForBone * 4 )
					lerp(array, (i, aFrame, bFrame, f) => {
						quat.slerp(quatFrames.subarray(i*4), aFrame.quat, bFrame.quat, f)
						quat.normalize(quatFrames.subarray(i*4), quatFrames.subarray(i*4))
					})
				}
			}
			{
				const array = fani.frameScaleArray
				if ( array.length ) {
					scaleFrames = new Float32Array( numFramesForBone * 3 )
					lerp(array, (i, aFrame, bFrame, f) => vec3.lerp(scaleFrames.subarray(i*3), aFrame.vec3, bFrame.vec3, f))
				}
			}
			{
				const array = fani.frameScaleUnkArray
				if ( array.length ) {
					if ( !scaleFrames ) {
						scaleFrames = new Float32Array( numFramesForBone * 3 )
						scaleFrames.fill(1)
					}
					console.log(scaleFrames)
					lerp(array, (i, aFrame, bFrame, f) => {
						const scale = aFrame.scale + (bFrame.scale - aFrame.scale) * f
						console.log(scale)
					})
				}
			}
			
			aniBonesNormalize.push({ posFrames, quatFrames, scaleFrames })
		})
		this.aniBones = aniBonesNormalize
	}
	
	_initOld(skeleton, aniData) {
		const MIN_FRAMES_DELTA = 160
		
		const aniBones = this.skeleton.bones.map(bone => {
			const ani = {
				framePosArray: [],
				frameQuatArray: [],
				frameScaleArray: [],
				frameScaleUnkArray: [],
			}
			const aniForBone = aniData.find(a => a.boneName === bone.boneName)
			if ( aniForBone ) {
				ani.framePosArray = aniForBone.framePosArray || []
				ani.frameQuatArray = aniForBone.frameQuatArray || []
				ani.frameScaleArray = aniForBone.frameUnkVec3Array || []
				ani.frameScaleUnkArray = aniForBone.frameScaleArray || []
			}
			return ani
		})
		
		const aniBoneToArr = fani => [fani.framePosArray, fani.frameQuatArray, fani.frameScaleArray, fani.frameScaleUnkArray]
		const deltaFrameList = []
		let maxFrame = 0
		aniBones.map(fani => {
			aniBoneToArr(fani).map(frs => {
				if ( !frs.length ) 
					return
				
				for(let i = 1; i < frs.length; i++) {
					const prev = frs[i-1]
					const next = frs[i]
					if ( prev.frame === next.frame ) {
						if ( JSON.stringify(prev) !== JSON.stringify(next) )
							throw new Error(`#1 Invalid animations...`)
						
						frs.splice(i, 1)
						i -= 2
					}
				}
				
				if ( frs.length < 2 )
					throw new Error(`#2 Invalid animations...`)
				
				if ( frs[0].frame !== 0 )
					throw new Error(`#3 Invalid animations...`)
				
				frs.shift()
				
				let prev = 0
				for(let i = 0; i < frs.length; i++) {
					const next = frs[i].frame
					const delta = next - prev
					prev = next
					
					deltaFrameList.push(delta)
				}
				maxFrame = Math.max(maxFrame, frs[frs.length - 1].frame)
			})			
		})
		deltaFrameList.sort((l, r) => l - r)
		if ( Math.min(...deltaFrameList) !== MIN_FRAMES_DELTA )
			throw new Error(`#4 Invalid animations...`)
		
		if ( deltaFrameList.every(v => v % MIN_FRAMES_DELTA !== 0) )
			throw new Error(`#5 Invalid animations...`)
		
		const tmpPos = [0,0,0], tmpPos2 = [0,0,0], tmpPos3 = [0,0,0]
		const numFramesForBone = maxFrame / MIN_FRAMES_DELTA
		const aniBonesNormalize = []
		aniBones.map(fani => {
			const lerp = (array, cb) => {
					for(let i = 0; i < array.length - 1; i++) {
						const rawFrame = array[i]
						const nextRawFrame = array[i + 1] || array[0]

						const index = rawFrame.frame / MIN_FRAMES_DELTA - 1
						const nextIndex = nextRawFrame.frame / MIN_FRAMES_DELTA - 1
						for(let j = index; j <= nextIndex; j++) {
							const f = 1 - (nextIndex - j) / (nextIndex - index)
							cb(j, rawFrame, nextRawFrame, f)
						}
					}
					
					const aFrame = array[ array.length - 1 ]
					const bFrame = array[0]
					const minFrameIndex = bFrame.frame / MIN_FRAMES_DELTA - 1
					for(let i = 0; i < minFrameIndex; i ++) {
						const f = (i + 1) / (minFrameIndex + 1)
						cb(i, aFrame, bFrame, f)
					}
				}
			
			let posFrames, quatFrames, scaleFrames
			{
				const array = fani.framePosArray
				if ( array.length ) {
					posFrames = new Float32Array( ( array[array.length - 1].frame / MIN_FRAMES_DELTA ) * 3 )
					lerp(array, (i, aFrame, bFrame, f) => vec3.lerp(posFrames.subarray(i*3), aFrame.pos, bFrame.pos, f))
				}
			}
			{
				const array = fani.frameQuatArray
				if ( array.length ) {
					fani.frameQuatArray.map(a => quat.invert(a.quat, a.quat))

					quatFrames = new Float32Array( ( array[array.length - 1].frame / MIN_FRAMES_DELTA ) * 4 )
					lerp(array, (i, aFrame, bFrame, f) => quat.slerp(quatFrames.subarray(i*4), aFrame.quat, bFrame.quat, f))
				}
			}
			{
				const array = fani.frameScaleArray
				if ( array.length ) {
					scaleFrames = new Float32Array( ( array[array.length - 1].frame / MIN_FRAMES_DELTA ) * 3 )
					lerp(array, (i, aFrame, bFrame, f) => vec3.lerp(scaleFrames.subarray(i*3), aFrame.vec3, bFrame.vec3, f))
				}
			}
			
			aniBonesNormalize.push({ posFrames, quatFrames, scaleFrames })
		})
		this.aniBones = aniBonesNormalize
	}
}
class Monster {
	constructor(skeleton, meshData, aniMap, url) {
		this.url = url
		this.skeleton = skeleton
		this.meshData = meshData
		this.aniMap = aniMap
		this.mesh = []
		this.glTextureMap = {}
		this.init()
	}

	static async load(url) {
		url = url.toLowerCase()
		const meshData = await (await fetch(url+"/meshs.json")).json()
		const bnData = await (await fetch(url+"/bones.json")).json()

		const skeleton = new Skeleton(bnData)
		if ( !skeleton.isValid )
			console.log(`Skeleton '${url}' invalid`)

		const aniList = await (await fetch(url+"/aniList.json")).json()
		const aniMap = (await Promise.all( aniList.map(async a => ({
			name: a,
			aniData: await (await fetch(url+"/" + a+'.json')).json()
		})) ) )
		.reduce((o, v) => ({
			...o, 
			[v.name]: new Animation(skeleton, v.aniData)
		}), {})
		
		return new this(skeleton, meshData, aniMap, url)
	}

	init() {
		[...new Set( this.meshData.map(m => m.textureName) )]
		.filter(v => v.length)
		.map(v => {
			this.glTextureMap[v] = glLoadTexture(this.url + '/textures/' + v.replace(/\.[^.]*$/, '.png'))
		})
		
		const mesh = this.mesh = []
		
		const boneVarinats = [], boneVarinatsMap = {}
		const vertices = []
		let vertexIndex = 0
		
		const meshData = this.meshData
			.filter(v => v.triangles.length)
			.sort((l, r) => l.textureName < r.textureName ? -1 : 1)
			.map(m => {
			
			const vertexStart = vertexIndex
			
			m.triangles.map(t => t.map(v => {
				vec3.transformMat4(v.vertex, v.vertex, m.matrix4x4_1)
				vec3.transformMat4(v.normal, v.normal, m.matrix4x4_1)
				
				const hash = v.boneInfoArray
					.sort((l, r) => l.boneName < r.boneName ? -1 : 1)
					.map(b => b.boneWeight+'/'+b.boneName)
					.join()
				
				if ( !boneVarinatsMap[hash] ) {
					const bonesData = v.boneInfoArray.map(b => {
						if ( this.skeleton.getBoneIndexByName(b.boneName) < 0 ) {
							//console.log(b.boneName)
						}
						return [this.skeleton.getBoneIndexByName(b.boneName), b.boneWeight]
					})
					boneVarinatsMap[hash] = boneVarinats.push( bonesData ) - 1
				}
				
				const matrixIndex =  boneVarinatsMap[hash]
				vertices.push(...v.vertex, ...v.normal, ...v.uv, matrixIndex)
				vertexIndex++
			}))

			const vertexCount = vertexIndex - vertexStart
			this.mesh.push({ 
				name: m.name,
				vertexStart, 
				vertexCount, 
				textureName: m.textureName,
				glTexture: this.glTextureMap[m.textureName]
			})
		})

		this.boneVarinats = boneVarinats
		this.vertices = new Float32Array(vertices)
		const start = glBuffer.alloc(this.vertices) / (3+3+2+1)
		this.mesh.map(m => m.vertexStart += start)
		
		this.glBufferVertices = glCreateBuffer(this.vertices)
		
		this.groupByTexture = [...new Set( mesh.map(m => m.textureName) )].map(textureName => {
			const sel = mesh.filter(m => m.textureName === textureName)
			const { vertexStart, glTexture } = sel[0]
			const vertexCount = sel.reduce((s, m) => s + m.vertexCount, 0)
			return { 
				vertexStart, 
				vertexCount,
				textureName, 
				glTexture, 
			}
		})
	}
}
class MonsterInstance {
	constructor(monster) {
		this.monster = monster
		this.skeletonInstance = new SkeletonInstance(this.monster.skeleton)
		
		this.boneMatrices = Array(this.monster.skeleton.bones.length).fill(0).map(v => mat4.create())
		this.matrices = Array(this.monster.boneVarinats.length).fill(0).map(v => mat4.create())

		const allocData = glFloatTextureMatrix.alloc(this.monster.boneVarinats.length)
		this.matrices = allocData.matrices
		this.matrixStart = allocData.start
		
		this.position = [0,0,0]
		this.rotateZ = 0
		this.scale = [1,1,1]
		
		this.timeStart = Date.now()
		
		this.aniEntries = Object.entries(this.monster.aniMap).map(v => [v[0].toLowerCase(), v[1]])
		
		this.loop()
	}
	
	calcMatrices() {
		this.skeletonInstance.identity()
		
		if ( 1 ) {
			const now = Date.now()
			const timeDelta = now - this.timeStart
			//let frameIndex = (timeDelta / 1000 * 160) | 0
			let frameIndex = (timeDelta / 1000) * gAniSpeed
			//console.log(frameIndex)
			let prevFrameIndex = frameIndex | 0
			let nextFrameIndex =  prevFrameIndex + 1
			let f = frameIndex - prevFrameIndex
			//prevFrameIndex = nextFrameIndex = (globalThis.aniFrameIndex = globalThis.aniFrameIndex ?? 25)
			let dv;
			const ani = this.aniEntries.find(v => ~v[0].indexOf(aniPattern))
			if ( ani ) {
				const { aniBones } = ani[1]
				for(let i = 0; i < aniBones.length; i++) {
					const aniBone = aniBones[i]
					const bone = this.skeletonInstance.bones[i]
					if ( aniBone.posFrames ) {
						dv = aniBone.posFrames.length / 3
						vec3.lerp( 
							bone.localPos,
							aniBone.posFrames.subarray( (prevFrameIndex % dv) * 3 ),
							aniBone.posFrames.subarray( (nextFrameIndex % dv) * 3 ),
							f
						)
					}
					
					if ( aniBone.quatFrames ) {
						const dv = aniBone.quatFrames.length / 4
						quat.slerp( 
							bone.localQuat,
							aniBone.quatFrames.subarray( (prevFrameIndex % dv) * 4 ),
							aniBone.quatFrames.subarray( (nextFrameIndex % dv) * 4 ),
							f
						)
						quat.normalize(bone.localQuat, bone.localQuat)
					}
					
					if ( aniBone.scaleFrames ) {
						const dv = aniBone.scaleFrames.length / 3
						quat.slerp( 
							bone.localScale,
							aniBone.scaleFrames.subarray( (prevFrameIndex % dv) * 3 ),
							aniBone.scaleFrames.subarray( (nextFrameIndex % dv) * 3 ),
							f
						)
					}
				}
				
			}
			//console.log(prevFrameIndex % dv, dv)
		}
		
		this.skeletonInstance.compose()
		
		const tmpMat4x4 = mat4.create()
		const summaryMat4x4 = mat4.create()
		for(let i = 0; i < this.matrices.length; i++) {
			const bone = this.skeletonInstance.bones[i]
			const boneVarinats = this.monster.boneVarinats[i]
			mat4.identity(summaryMat4x4)
			
			let first = 0
			
			for(const [boneIndex, boneWeight] of boneVarinats) {
				if ( boneIndex < 0 ) continue
				
				const { finalMat4x4 } = this.skeletonInstance.bones[ boneIndex ]
				
				//mat4.multiplyScalar(tmpMat4x4, finalMat4x4, 1)
				//mat4.copy(summaryMat4x4, finalMat4x4)
				//break
				
				mat4.multiplyScalar(tmpMat4x4, finalMat4x4, boneWeight)
				if ( !first++ ) {
					mat4.copy(summaryMat4x4, finalMat4x4)
				} else {
					mat4.add(summaryMat4x4, tmpMat4x4, summaryMat4x4)
				}
			}
		
			mat4.copy(this.matrices[i], summaryMat4x4)
		}
	}
	
	_tmpMatrixTranslation = new Float32Array(16)
	_tmpMatrixRotation = new Float32Array(16)
	_tmpMatrixScale = new Float32Array(16)
	_tmpMatrix = new Float32Array(16)
	loop() {
		this.calcMatrices()

		mat4.fromScaling(this._tmpMatrixScale, this.scale)
		mat4.fromTranslation(this._tmpMatrixTranslation, this.position)
		mat4.fromZRotation(this._tmpMatrixRotation, this.rotateZ)
		
		mat4.multiply(this._tmpMatrix, this._tmpMatrixRotation, this._tmpMatrixScale)
		mat4.multiply(this._tmpMatrix, this._tmpMatrixTranslation, this._tmpMatrix)
		
		mat4.multiply(this._tmpMatrix, modelViewMatrix, this._tmpMatrix)
		
		

		const matrices = glFloatTextureMatrix.getMemory(this.matrixStart)
		
		for(let i = 0; i < this.matrices.length; i++) {
			const matrix = this.matrices[i]
			mat4.multiply(matrices.subarray(i*16, i*16 + 16), this._tmpMatrix, matrix)
		}
	}
	draw() {
		/*
		const type = gl.FLOAT;
		const normalize = false;
		const stride = (3 + 3 + 2 + 1)*4
		const offset = 0;
			
		gl.bindBuffer(gl.ARRAY_BUFFER, this.monster.glBufferVertices)
		gl.vertexAttribPointer( programInfo.attribLocations.aVertexPosition, 3, type, normalize, stride, 0 )
		gl.vertexAttribPointer( programInfo.attribLocations.aNormal        , 3, type, normalize, stride, (3)*4 )
		gl.vertexAttribPointer( programInfo.attribLocations.aUV            , 2, type, normalize, stride, (3+3)*4 )
		gl.vertexAttribPointer( programInfo.attribLocations.aMatrixIndex   , 1, type, normalize, stride, (3+3+2)*4 )	
		*/
		
		
		if( 0 )
		this.monster.groupByTexture.map(mesh => {
			glSetTexture(programInfo.uniformLocations.uTextureDiff, 1, mesh.glTexture)
			gl.drawArrays(gl.TRIANGLES, mesh.vertexStart, mesh.vertexCount)
			//glExtInstancedArrays.drawArraysInstancedANGLE(gl.TRIANGLES, m.start, m.length, 1)
		})
		
		if(0)
		this.monster.mesh.map(mesh => {
			glSetTexture(programInfo.uniformLocations.uTextureDiff, 1, mesh.glTexture)
			gl.drawArrays(gl.TRIANGLES, mesh.vertexStart, mesh.vertexCount)
		})
		
		this.monster.meshData.map(rawMesh => {
			const mesh = this.monster.mesh.find(m => m.name === rawMesh.name)
			if ( !mesh )
				return
			
			glSetTexture(programInfo.uniformLocations.uTextureDiff, 1, mesh.glTexture)
			gl.drawArrays(gl.TRIANGLES, mesh.vertexStart, mesh.vertexCount)
		})
	}
}

Array.prototype.getFirstItem = function() {
	return this[ 0 ]
}	
Array.prototype.getLastItem = function() {
	return this[ this.length - 1 ]
}	

function goFrame(frame) {
	frame %= Math.max(...aniData.map(v => v.maxFrame))
	aniData
		.filter(v => v.readlNumFrame > 10)
		.map(ani => {
			const bone = skeleton.bones.find(b => b.name === ani.boneName)
			if ( !bone )
				return
			
			let pos, localQuat, localScale, localScaleOne
			
			pos = ani.framePosArray.find(v => v.frame >= frame)
			localQuat = ani.frameQuatArray.find(v => v.frame >= frame)
			localScale = ani.frameUnkVec3Array.find(v => v.frame >= frame)
			localScaleOne = ani.frameScaleArray.find(v => v.frame >= frame)
			
			if( 1 )
			if ( pos ) {
				bone.localPos[0] = pos.pos[0]
				bone.localPos[1] = pos.pos[1]
				bone.localPos[2] = pos.pos[2]
			}
			
			if( 1 )
			if ( localQuat ) {
				localQuat = objToArray(localQuat.quat)
				//console.log(localQuat)
				quat.normalize(localQuat, localQuat)
				quat.invert(localQuat, localQuat)
				quat.copy(bone.localQuat, localQuat)
			}
			
			if ( 1 )
			if ( localScale ) {
				quat.copy(bone.localScale, localScale)
			}
			
			if ( 1 )
			if ( localScaleOne ) {
				quat.copy(bone.localScale, [localScaleOne.scale, localScaleOne.scale, localScaleOne.scale])
			}
			
		})
	
	skeleton.calcMatrix()
	skeleton.calcGlobalMatrix()
}

globalThis.rY=0
//setInterval(() => rY += 1, 20)
const fps = Fps()
let loopCount = 0
function loop() {
	loopCount++
	
	clear()
	
	loopView()
	
	let t = performance.now()
	monsters.map(m => m.loop())
	t = performance.now() - t
	//console.log(1000/(t))
	
	glFloatTextureMatrix.floatTexture.swap()
	gl.uniform2f(programInfo.uniformLocations.uTextureFloatMatrixSIZE, glFloatTextureMatrix.floatTexture.width, glFloatTextureMatrix.floatTexture.height)
	
	draw()
	draw2( 0 )
	
	fps()
	
	numObjectsLoop()
	
	requestAnimationFrame(loop)
}

function objToArray(obj) {
	return Object.keys(obj).map(i => obj[i])
}

function glCreateBuffer(data) {
	const glBuffer = gl.createBuffer()
	gl.bindBuffer(gl.ARRAY_BUFFER, glBuffer)
	gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW)
	return glBuffer
}

let monsterClassList = []
function addRndMonster() {
	const dist = createDistMonsters
	const mi = new MonsterInstance( monsterClassList[ Math.random() * monsterClassList.length |0 ] )
	mi.position[0] = (Math.random() - 0.5) * dist
	mi.position[1] = (Math.random() - 0.5) * dist
	mi.position[2] = (Math.random() - 0.5) * dist * 0
	monsters.push(mi)
	console.log('added monster')
}

const hideMonsters = []
let numMonsters = 0
function numObjectsLoop() {
	const _numMonsters = forceNumMonsters || numMonsters
	
	const needUpdate = _numMonsters !== monsters.length
	let maxAddNum = 1
	while( _numMonsters > monsters.length && hideMonsters.length && maxAddNum-- > 0 ) monsters.push( hideMonsters.pop() )
	while( _numMonsters > monsters.length && maxAddNum-- > 0 ) addRndMonster()
	while( _numMonsters < monsters.length && monsters.length ) hideMonsters.push( monsters.pop() )
	if ( needUpdate )	
		postMessage({action: 'informNumObjects', data: monsters.length})
}

let vertexData, glVertexBuffer, glTextureDiff, glTextureDiffMap = {}
async function init(canvas) {
	main(canvas)
	
	viewPort.init(canvas, gl)

    const glExtTextureFloat = gl.getExtension('OES_texture_float')
    if ( !glExtTextureFloat )
		return alert("No support for OES_texture_float")
    
 
	const glExtInstancedArrays = gl.getExtension('ANGLE_instanced_arrays');
	if ( !glExtInstancedArrays ) {
		return alert('need ANGLE_instanced_arrays');
	}
	globalThis.glExtInstancedArrays=glExtInstancedArrays
	
	const glExtTextureFilterAnisotropic = (
		gl.getExtension('EXT_texture_filter_anisotropic') ||
		gl.getExtension('MOZ_EXT_texture_filter_anisotropic') ||
		gl.getExtension('WEBKIT_EXT_texture_filter_anisotropic')
	)
	globalThis.glExtTextureFilterAnisotropic=glExtTextureFilterAnisotropic
	if ( glExtTextureFilterAnisotropic ) {
		glExtTextureFilterAnisotropic.maxAnisotropy = gl.getParameter(glExtTextureFilterAnisotropic.MAX_TEXTURE_MAX_ANISOTROPY_EXT);
	}	

	glFloatTextureMatrix.init()
	glBufferInstancedFloat.init()	

	monsterClassList = [
		/*
		await Monster.load('tmp/BLOODYSASSASSIN'),
		await Monster.load('tmp/callianaarcher'),
		await Monster.load('tmp/FLEM'),
		await Monster.load('tmp/ANABOLA'),
		await Monster.load('tmp/OPSLAVA'),
		await Monster.load('tmp/MATY'),
		await Monster.load('tmp/ARGHOL'),
		*/
		/*
		await Monster.load('tmp/ANABOLA'),
		await Monster.load('tmp/callianaarcher'),
		await Monster.load('tmp/TOYFELLSTRIKER'),
		await Monster.load('tmp/HIGHELFARCHMAGE'),
		await Monster.load('tmp/FLEM'),
		await Monster.load('tmp/ARGHOL'),
		await Monster.load('tmp/CLOD'),
		//await Monster.load('tmp/HOLYSTONEKEEPER'),
		await Monster.load('tmp/VARAS'),
		await Monster.load('tmp/BRUTAL'),
		await Monster.load('tmp/BOGYBOLT'),
		await Monster.load('tmp/TRESHU'),
		await Monster.load('tmp/GLADIUS'),
		await Monster.load('tmp/KLAN'),
		await Monster.load('tmp/OPSLAVA'),
		await Monster.load('tmp/MATY'),
		await Monster.load('tmp/BLOODYSASSASSIN'),
		
		await Monster.load('tmp/HEAVYWING'),
		await Monster.load('tmp/KILZAICHEF'),
		*/
		await Monster.load('tmp/TURNCOATMILER'),
	]
	
	globalThis.monsters = []
	
	globalThis.matrixFloatTexture = new FloatTexture()

	loop()
}

globalThis.addEventListener('message', ({data: { action, data }}) => {
	switch(action) {
		case "canvas":
			if ( typeof data === 'string' )
				data = globalThis[data]
			init(data)
			break
		
		case "loadTextureResult":
			loadTexturesMap[data.rpcID](data.imgData)
			break
			
		case 'setNumObjects':
			numMonsters = data
			break
			
		case 'mousemove':
			camera.mousemove(data)
			break
		
		case 'setAniPattern':
			aniPattern = data 
			break
		
		case 'resize':
			viewPort.update(data)
			break
		
		case 'setSettings':
			data = { forceNumMonsters, zDist, createDistMonsters, gAniSpeed, ...data };
			forceNumMonsters = +data.forceNumMonsters
			zDist = +data.zDist
			gAniSpeed = +data.gAniSpeed
			createDistMonsters = +data.createDistMonsters
			camera.update(data)
			break
		
	}
})