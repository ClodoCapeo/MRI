/**
 * Parser for Analyze 7.5 format (.hdr/.img pairs)
 * Based on the format specification: http://www.grahamwideman.com/gw/brain/analyze/formatdoc.htm
 */

export interface AnalyzeHeader {
  dimensions: number[]
  dataType: number
  pixelDimensions: number[]
  voxelOffset: number
  dataTypeString: string
  littleEndian: boolean
}

export interface AnalyzeVolume {
  header: AnalyzeHeader
  data: Float32Array | Int16Array | Uint8Array
  min: number
  max: number
}

export async function parseAnalyzeHeader(headerBuffer: ArrayBuffer): Promise<AnalyzeHeader> {
  // Check endianness
  const view = new DataView(headerBuffer)
  const sizeof_hdr = view.getInt32(0, true) // Try little endian first
  const littleEndian = sizeof_hdr === 348 // If correct, this should be 348

  // Parse dimensions
  const dimensions = []
  const dim0 = view.getInt16(40, littleEndian)
  for (let i = 0; i < dim0; i++) {
    dimensions.push(view.getInt16(40 + 2 * (i + 1), littleEndian))
  }

  // Parse data type
  const dataType = view.getInt16(70, littleEndian)
  let dataTypeString = "unknown"

  switch (dataType) {
    case 2:
      dataTypeString = "uint8"
      break
    case 4:
      dataTypeString = "int16"
      break
    case 8:
      dataTypeString = "int32"
      break
    case 16:
      dataTypeString = "float32"
      break
    case 64:
      dataTypeString = "float64"
      break
  }

  // Parse voxel dimensions
  const pixelDimensions = []
  for (let i = 0; i < dim0; i++) {
    pixelDimensions.push(view.getFloat32(80 + 4 * (i + 1), littleEndian))
  }

  // Get voxel offset
  const voxelOffset = view.getFloat32(108, littleEndian)

  return {
    dimensions,
    dataType,
    pixelDimensions,
    voxelOffset,
    dataTypeString,
    littleEndian,
  }
}

export async function parseAnalyzeVolume(headerBuffer: ArrayBuffer, imageBuffer: ArrayBuffer): Promise<AnalyzeVolume> {
  const header = await parseAnalyzeHeader(headerBuffer)
  const { dimensions, dataType, littleEndian } = header

  // Calculate total number of voxels
  const totalVoxels = dimensions.reduce((a, b) => a * b, 1)

  // Create appropriate typed array based on data type
  let data: Float32Array | Int16Array | Uint8Array
  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY

  if (dataType === 2) {
    // uint8
    data = new Uint8Array(imageBuffer)
  } else if (dataType === 4) {
    // int16
    data = new Int16Array(totalVoxels)
    const view = new DataView(imageBuffer)
    for (let i = 0; i < totalVoxels; i++) {
      const value = view.getInt16(i * 2, littleEndian)
      data[i] = value
      min = Math.min(min, value)
      max = Math.max(max, value)
    }
  } else if (dataType === 16) {
    // float32
    data = new Float32Array(totalVoxels)
    const view = new DataView(imageBuffer)
    for (let i = 0; i < totalVoxels; i++) {
      const value = view.getFloat32(i * 4, littleEndian)
      data[i] = value
      min = Math.min(min, value)
      max = Math.max(max, value)
    }
  } else {
    throw new Error(`Unsupported data type: ${dataType}`)
  }

  // For uint8, calculate min/max
  if (dataType === 2) {
    for (let i = 0; i < totalVoxels; i++) {
      min = Math.min(min, data[i])
      max = Math.max(max, data[i])
    }
  }

  return {
    header,
    data,
    min,
    max,
  }
}

// Extract a 2D slice from a 3D volume
export function extractSlice(
  volume: AnalyzeVolume,
  sliceIndex: number,
  axis: 0 | 1 | 2 = 2,
): { data: Float32Array; width: number; height: number } {
  const { dimensions, dataTypeString } = volume.header
  const [width, height, depth] = dimensions

  let sliceWidth: number
  let sliceHeight: number
  let sliceData: Float32Array

  if (axis === 0) {
    // Sagittal slice (YZ plane)
    sliceWidth = height
    sliceHeight = depth
    sliceData = new Float32Array(sliceWidth * sliceHeight)

    for (let y = 0; y < sliceHeight; y++) {
      for (let x = 0; x < sliceWidth; x++) {
        const volumeIndex = sliceIndex + x * width + y * width * height
        sliceData[x + y * sliceWidth] = volume.data[volumeIndex]
      }
    }
  } else if (axis === 1) {
    // Coronal slice (XZ plane)
    sliceWidth = width
    sliceHeight = depth
    sliceData = new Float32Array(sliceWidth * sliceHeight)

    for (let y = 0; y < sliceHeight; y++) {
      for (let x = 0; x < sliceWidth; x++) {
        const volumeIndex = x + sliceIndex * width + y * width * height
        sliceData[x + y * sliceWidth] = volume.data[volumeIndex]
      }
    }
  } else {
    // Axial slice (XY plane)
    sliceWidth = width
    sliceHeight = height
    sliceData = new Float32Array(sliceWidth * sliceHeight)

    for (let y = 0; y < sliceHeight; y++) {
      for (let x = 0; x < sliceWidth; x++) {
        const volumeIndex = x + y * width + sliceIndex * width * height
        sliceData[x + y * sliceWidth] = volume.data[volumeIndex]
      }
    }
  }

  return {
    data: sliceData,
    width: sliceWidth,
    height: sliceHeight,
  }
}

// Normalize slice data to [0, 1] range
export function normalizeSlice(slice: Float32Array, min?: number, max?: number): Float32Array {
  const result = new Float32Array(slice.length)

  // Calculate min/max if not provided
  if (min === undefined || max === undefined) {
    min = Number.POSITIVE_INFINITY
    max = Number.NEGATIVE_INFINITY
    for (let i = 0; i < slice.length; i++) {
      min = Math.min(min!, slice[i])
      max = Math.max(max!, slice[i])
    }
  }

  // Avoid division by zero
  const range = max! - min!
  const scale = range === 0 ? 0 : 1 / range

  // Normalize
  for (let i = 0; i < slice.length; i++) {
    result[i] = (slice[i] - min!) * scale
  }

  return result
}

// Convert slice data to ImageData for canvas rendering
export function sliceToImageData(
  slice: { data: Float32Array; width: number; height: number },
  colormap?: (value: number) => [number, number, number, number],
): ImageData {
  const { data, width, height } = slice
  const imageData = new ImageData(width, height)

  // Default grayscale colormap
  const defaultColormap = (value: number): [number, number, number, number] => {
    const v = Math.floor(value * 255)
    return [v, v, v, 255]
  }

  const cmap = colormap || defaultColormap

  for (let i = 0; i < data.length; i++) {
    const [r, g, b, a] = cmap(data[i])
    imageData.data[i * 4] = r
    imageData.data[i * 4 + 1] = g
    imageData.data[i * 4 + 2] = b
    imageData.data[i * 4 + 3] = a
  }

  return imageData
}

// Segmentation colormap
export function segmentationColormap(value: number): [number, number, number, number] {
  // iSeg-2017 has 4 classes:
  // 0: Background
  // 1: CSF (Cerebrospinal fluid)
  // 2: GM (Gray matter)
  // 3: WM (White matter)
  switch (Math.round(value * 3)) {
    case 0:
      return [0, 0, 0, 0] // Background (transparent)
    case 1:
      return [65, 105, 225, 200] // CSF (royal blue, semi-transparent)
    case 2:
      return [50, 205, 50, 200] // GM (lime green, semi-transparent)
    case 3:
      return [255, 165, 0, 200] // WM (orange, semi-transparent)
    default:
      return [255, 0, 0, 200] // Error (red, semi-transparent)
  }
}
