import * as tf from "@tensorflow/tfjs"
import type { AnalyzeVolume } from "./analyze-parser"

export class DataGenerator {
  constructor(
    private volumes: { t1: AnalyzeVolume; t2: AnalyzeVolume; groundTruth?: AnalyzeVolume }[],
    private sliceRange: [number, number] = [60, 135], // As suggested in tutorial
    private targetSize: [number, number] = [128, 128], // As suggested in tutorial
  ) {}

  async *generate(batchSize = 4) {
    if (this.volumes.length === 0) {
      console.warn("No volumes provided to DataGenerator")
      return
    }

    try {
      // Ensure TensorFlow.js is ready
      await tf.ready()

      // Shuffle volumes
      const shuffledIndices = Array.from({ length: this.volumes.length }, (_, i) => i)
      for (let i = shuffledIndices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        ;[shuffledIndices[i], shuffledIndices[j]] = [shuffledIndices[j], shuffledIndices[i]]
      }

      // Generate batches
      for (let i = 0; i < shuffledIndices.length; i += batchSize) {
        const batchIndices = shuffledIndices.slice(i, i + batchSize)
        const X: tf.Tensor4D[] = []
        const Y: tf.Tensor4D[] = []

        for (const idx of batchIndices) {
          const volume = this.volumes[idx]
          if (!volume || !volume.t1 || !volume.t2) {
            console.warn(`Invalid volume at index ${idx}, skipping`)
            continue
          }

          // Process slices in the specified range
          const maxDepth = Math.min(
            volume.t1.header.dimensions[2],
            volume.t2.header.dimensions[2],
            volume.groundTruth ? volume.groundTruth.header.dimensions[2] : Number.POSITIVE_INFINITY,
          )

          const startSlice = Math.min(this.sliceRange[0], maxDepth - 1)
          const endSlice = Math.min(this.sliceRange[1], maxDepth)

          for (let z = startSlice; z < endSlice; z++) {
            try {
              // Extract and process T1 slice
              const t1Data = this.extractSlice(volume.t1, z)
              const t1Tensor = tf.tensor2d(t1Data.data, [t1Data.height, t1Data.width])
              const resizedT1 = tf.image.resizeBilinear(t1Tensor.expandDims(2), this.targetSize).squeeze()

              // Extract and process T2 slice
              const t2Data = this.extractSlice(volume.t2, z)
              const t2Tensor = tf.tensor2d(t2Data.data, [t2Data.height, t2Data.width])
              const resizedT2 = tf.image.resizeBilinear(t2Tensor.expandDims(2), this.targetSize).squeeze()

              // Combine slices
              const combinedSlice = tf.stack([resizedT1, resizedT2], 2).expandDims(0)
              X.push(combinedSlice)

              // Process ground truth if available
              if (volume.groundTruth) {
                const gtData = this.extractSlice(volume.groundTruth, z)
                const gtTensor = tf.tensor2d(gtData.data, [gtData.height, gtData.width])

                // Convert class 4 to class 3 as in tutorial
                const gtFixed = tf.where(gtTensor.equal(tf.scalar(4)), tf.scalar(3), gtTensor)

                // One-hot encode
                const oneHot = tf.oneHot(tf.cast(gtFixed, "int32"), 4)

                // Resize
                const resizedGt = tf.image.resizeBilinear(oneHot.expandDims(0), this.targetSize)
                Y.push(resizedGt)

                // Clean up
                gtTensor.dispose()
                gtFixed.dispose()
              }

              // Clean up
              t1Tensor.dispose()
              t2Tensor.dispose()
              resizedT1.dispose()
              resizedT2.dispose()
            } catch (error) {
              console.error(`Error processing slice ${z} for volume ${idx}:`, error)
            }
          }
        }

        if (X.length === 0) continue

        // Concatenate batch tensors
        const batchX = tf.concat(X, 0)

        // Normalize to [0,1]
        const normalizedX = batchX.div(batchX.max())

        if (Y.length > 0 && Y.length === X.length) {
          const batchY = tf.concat(Y, 0)
          yield {
            xs: normalizedX,
            ys: batchY,
          }
          batchY.dispose()
        } else {
          yield {
            xs: normalizedX,
            ys: undefined,
          }
        }

        // Clean up
        batchX.dispose()
        normalizedX.dispose()
      }
    } catch (error) {
      console.error("Error in data generator:", error)
      throw error
    }
  }

  private extractSlice(
    volume: AnalyzeVolume,
    sliceIndex: number,
  ): { data: Float32Array; width: number; height: number } {
    const { dimensions } = volume.header
    const [width, height] = dimensions

    // Create a slice buffer
    const sliceData = new Float32Array(width * height)

    // Extract the slice data
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const volumeIndex = x + y * width + sliceIndex * width * height
        sliceData[x + y * width] = volume.data[volumeIndex]
      }
    }

    // Normalize slice data to [0,1]
    let min = Number.POSITIVE_INFINITY
    let max = Number.NEGATIVE_INFINITY

    for (let i = 0; i < sliceData.length; i++) {
      min = Math.min(min, sliceData[i])
      max = Math.max(max, sliceData[i])
    }

    const range = max - min
    if (range > 0) {
      for (let i = 0; i < sliceData.length; i++) {
        sliceData[i] = (sliceData[i] - min) / range
      }
    }

    return {
      data: sliceData,
      width,
      height,
    }
  }
}
