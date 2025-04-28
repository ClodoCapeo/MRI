import * as tf from "@tensorflow/tfjs"
import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

// Utility function for combining class names
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Calculate Dice coefficient between prediction and ground truth
export function calculateDiceCoefficient(prediction: tf.Tensor, groundTruth: tf.Tensor): number {
  return tf.tidy(() => {
    // Ensure both tensors have the same shape
    if (!tf.util.arraysEqual(prediction.shape, groundTruth.shape)) {
      throw new Error("Prediction and ground truth must have the same shape")
    }

    // Calculate Dice coefficient for each class and average
    let totalDice = 0
    let classCount = 0

    for (let classIndex = 1; classIndex < 4; classIndex++) {
      // Create binary masks for the current class
      const predMask = prediction.equal(tf.scalar(classIndex, "int32"))
      const gtMask = groundTruth.equal(tf.scalar(classIndex, "int32"))

      // Calculate intersection and union
      const intersection = predMask.logicalAnd(gtMask).sum()
      const union = predMask.sum().add(gtMask.sum())

      // Calculate Dice coefficient: 2 * intersection / union
      const dice = intersection.mul(2).div(union).dataSync()[0]

      // Only count classes that are present in the ground truth
      if (!isNaN(dice)) {
        totalDice += dice
        classCount++
      }
    }

    // Return average Dice coefficient
    return classCount > 0 ? totalDice / classCount : 0
  })
}

// Convert tensor to canvas for visualization
export function tensorToCanvas(tensor: tf.Tensor, canvas: HTMLCanvasElement, isSegmentation = false): void {
  const ctx = canvas.getContext("2d")
  if (!ctx) return

  tf.tidy(() => {
    // Get tensor dimensions
    const [height, width] = tensor.shape

    // Resize canvas to match tensor dimensions
    canvas.width = width
    canvas.height = height

    // Create ImageData
    const imageData = ctx.createImageData(width, height)
    const data = imageData.data

    // Get tensor data
    const tensorData = tensor.dataSync()

    if (isSegmentation) {
      // Colormap for segmentation
      for (let i = 0; i < height * width; i++) {
        const classIndex = Math.round(tensorData[i])

        // Apply colormap based on class
        switch (classIndex) {
          case 0: // Background
            data[i * 4] = 0
            data[i * 4 + 1] = 0
            data[i * 4 + 2] = 0
            data[i * 4 + 3] = 0
            break
          case 1: // CSF
            data[i * 4] = 65
            data[i * 4 + 1] = 105
            data[i * 4 + 2] = 225
            data[i * 4 + 3] = 200
            break
          case 2: // GM
            data[i * 4] = 50
            data[i * 4 + 1] = 205
            data[i * 4 + 2] = 50
            data[i * 4 + 3] = 200
            break
          case 3: // WM
            data[i * 4] = 255
            data[i * 4 + 1] = 165
            data[i * 4 + 2] = 0
            data[i * 4 + 3] = 200
            break
          default:
            data[i * 4] = 255
            data[i * 4 + 1] = 0
            data[i * 4 + 2] = 0
            data[i * 4 + 3] = 200
        }
      }
    } else {
      // Normalize to [0, 255] for grayscale
      const min = tensor.min().dataSync()[0]
      const max = tensor.max().dataSync()[0]
      const range = max - min

      for (let i = 0; i < height * width; i++) {
        const value = Math.round(((tensorData[i] - min) / range) * 255)
        data[i * 4] = value
        data[i * 4 + 1] = value
        data[i * 4 + 2] = value
        data[i * 4 + 3] = 255
      }
    }

    ctx.putImageData(imageData, 0, 0)
  })
}

export async function loadSampleData(): Promise<{
  t1: HTMLImageElement
  t2: HTMLImageElement
  groundTruth: HTMLImageElement
}> {
  return new Promise((resolve, reject) => {
    const t1 = new Image()
    t1.src = "/sample-data/subject-1-t1.png"
    t1.onload = () => {
      const t2 = new Image()
      t2.src = "/sample-data/subject-1-t2.png"
      t2.onload = () => {
        const groundTruth = new Image()
        groundTruth.src = "/sample-data/subject-1-label.png"
        groundTruth.onload = () => {
          resolve({ t1, t2, groundTruth })
        }
        groundTruth.onerror = reject
      }
      t2.onerror = reject
    }
    t1.onerror = reject
  })
}
