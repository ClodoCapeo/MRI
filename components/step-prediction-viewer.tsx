"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent } from "@/components/ui/card"
import * as tf from "@tensorflow/tfjs"
import type { AnalyzeVolume } from "@/lib/analyze-parser"
import type { EnhancedUNet } from "@/lib/unet-enhanced"

interface StepPredictionViewerProps {
  step: number
  t1Volume: AnalyzeVolume | null
  t2Volume: AnalyzeVolume | null
  groundTruthVolume: AnalyzeVolume | null
  unetModel: EnhancedUNet | null
}

export default function StepPredictionViewer({
  step,
  t1Volume,
  t2Volume,
  groundTruthVolume,
  unetModel,
}: StepPredictionViewerProps) {
  const [predictionImage, setPredictionImage] = useState<string | null>(null)
  const [groundTruthImage, setGroundTruthImage] = useState<string | null>(null)
  const [diceCoefficient, setDiceCoefficient] = useState<number | null>(null)
  const [sliceIndex, setSliceIndex] = useState<number>(0)
  const [isProcessing, setIsProcessing] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const gtCanvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!t1Volume || !t2Volume || !unetModel) return

    const generatePrediction = async () => {
      setIsProcessing(true)
      setError(null)

      try {
        // Get middle slice if not set
        if (sliceIndex === 0) {
          const depth = t1Volume.header.dimensions[2]
          setSliceIndex(Math.floor(depth / 2))
          return // Will re-run when sliceIndex is updated
        }

        // Convert volumes to tensors
        const t1Tensor = tf.tensor3d(Array.from(t1Volume.data), t1Volume.header.dimensions)
        const t2Tensor = tf.tensor3d(Array.from(t2Volume.data), t2Volume.header.dimensions)

        // Normalize tensors
        const t1Normalized = t1Tensor.sub(t1Tensor.min()).div(t1Tensor.max().sub(t1Tensor.min()))
        const t2Normalized = t2Tensor.sub(t2Tensor.min()).div(t2Tensor.max().sub(t2Tensor.min()))

        // Get prediction based on step
        const prediction: tf.Tensor3D | null = null

        if (step === 1 && unetModel.modelStep1) {
          // For step 1, we only need to predict brain vs background
          const [width, height, depth] = t1Volume.header.dimensions
          const slices: tf.Tensor3D[] = []

          // Extract the slice
          const t1Slice = t1Normalized.slice([0, 0, sliceIndex], [width, height, 1])
          const t2Slice = t2Normalized.slice([0, 0, sliceIndex], [width, height, 1])

          // Resize to 128x128
          const resizedT1 = tf.image.resizeBilinear(t1Slice, [128, 128])
          const resizedT2 = tf.image.resizeBilinear(t2Slice, [128, 128])

          // Combine slices
          const combined = tf.concat([resizedT1, resizedT2], 2).expandDims(0)

          // Predict
          const stepPrediction = unetModel.modelStep1.predict(combined) as tf.Tensor4D
          const binaryPrediction = stepPrediction.greater(0.5).toFloat().squeeze(0)

          // Resize back to original dimensions
          const resizedPrediction = tf.image.resizeBilinear(binaryPrediction, [width, height])

          // Render to canvas
          renderToCanvas(resizedPrediction.squeeze(), canvasRef.current!, false)

          // Calculate Dice coefficient if ground truth is available
          if (groundTruthVolume) {
            const gtTensor = tf.tensor3d(Array.from(groundTruthVolume.data), groundTruthVolume.header.dimensions)
            const gtNormalized = gtTensor.sub(gtTensor.min()).div(gtTensor.max().sub(gtTensor.min()))
            const gtSlice = gtNormalized.slice([0, 0, sliceIndex], [width, height, 1])

            // Create binary mask for ground truth (any value > 0 is brain)
            const gtBinary = gtSlice.greater(0).toFloat()

            // Render ground truth to canvas
            renderToCanvas(gtBinary.squeeze(), gtCanvasRef.current!, false)

            // Calculate Dice coefficient
            const dice = await calculateDice(resizedPrediction.squeeze(), gtBinary.squeeze())
            setDiceCoefficient(dice)
          }

          // Clean up
          t1Slice.dispose()
          t2Slice.dispose()
          resizedT1.dispose()
          resizedT2.dispose()
          combined.dispose()
          stepPrediction.dispose()
          binaryPrediction.dispose()
          resizedPrediction.dispose()
        } else if (step === 2 && unetModel.modelStep1 && unetModel.modelStep2) {
          // For step 2, we need to use the output of step 1
          const [width, height, depth] = t1Volume.header.dimensions

          // Extract the slice
          const t1Slice = t1Normalized.slice([0, 0, sliceIndex], [width, height, 1])
          const t2Slice = t2Normalized.slice([0, 0, sliceIndex], [width, height, 1])

          // Resize to 128x128
          const resizedT1 = tf.image.resizeBilinear(t1Slice, [128, 128])
          const resizedT2 = tf.image.resizeBilinear(t2Slice, [128, 128])

          // Get step 1 prediction
          const step1Input = tf.concat([resizedT1, resizedT2], 2).expandDims(0)
          const step1Prediction = unetModel.modelStep1.predict(step1Input) as tf.Tensor4D
          const step1Binary = step1Prediction.greater(0.5).toFloat().squeeze(0)

          // Combine for step 2
          const step2Input = tf.concat([resizedT1, resizedT2, step1Binary], 2).expandDims(0)

          // Predict
          const step2Prediction = unetModel.modelStep2.predict(step2Input) as tf.Tensor4D
          const binaryPrediction = step2Prediction.greater(0.5).toFloat().squeeze(0)

          // Resize back to original dimensions
          const resizedPrediction = tf.image.resizeBilinear(binaryPrediction, [width, height])

          // Render to canvas
          renderToCanvas(resizedPrediction.squeeze(), canvasRef.current!, false)

          // Calculate Dice coefficient if ground truth is available
          if (groundTruthVolume) {
            const gtTensor = tf.tensor3d(Array.from(groundTruthVolume.data), groundTruthVolume.header.dimensions)
            const gtNormalized = gtTensor.sub(gtTensor.min()).div(gtTensor.max().sub(gtTensor.min()))
            const gtSlice = gtNormalized.slice([0, 0, sliceIndex], [width, height, 1])

            // Create binary mask for ground truth (values > 10 are white matter)
            const gtBinary = gtSlice.greater(10).toFloat()

            // Render ground truth to canvas
            renderToCanvas(gtBinary.squeeze(), gtCanvasRef.current!, false)

            // Calculate Dice coefficient
            const dice = await calculateDice(resizedPrediction.squeeze(), gtBinary.squeeze())
            setDiceCoefficient(dice)
          }

          // Clean up
          t1Slice.dispose()
          t2Slice.dispose()
          resizedT1.dispose()
          resizedT2.dispose()
          step1Input.dispose()
          step1Prediction.dispose()
          step1Binary.dispose()
          step2Input.dispose()
          step2Prediction.dispose()
          binaryPrediction.dispose()
          resizedPrediction.dispose()
        } else if (step === 3 && unetModel.modelStep1 && unetModel.modelStep2 && unetModel.modelStep3) {
          // For step 3, we need outputs from steps 1 and 2
          const [width, height, depth] = t1Volume.header.dimensions

          // Extract the slice
          const t1Slice = t1Normalized.slice([0, 0, sliceIndex], [width, height, 1])
          const t2Slice = t2Normalized.slice([0, 0, sliceIndex], [width, height, 1])

          // Resize to 128x128
          const resizedT1 = tf.image.resizeBilinear(t1Slice, [128, 128])
          const resizedT2 = tf.image.resizeBilinear(t2Slice, [128, 128])

          // Get step 1 prediction
          const step1Input = tf.concat([resizedT1, resizedT2], 2).expandDims(0)
          const step1Prediction = unetModel.modelStep1.predict(step1Input) as tf.Tensor4D
          const step1Binary = step1Prediction.greater(0.5).toFloat().squeeze(0)

          // Get step 2 prediction
          const step2Input = tf.concat([resizedT1, resizedT2, step1Binary], 2).expandDims(0)
          const step2Prediction = unetModel.modelStep2.predict(step2Input) as tf.Tensor4D
          const step2Binary = step2Prediction.greater(0.5).toFloat().squeeze(0)

          // Combine for step 3
          const step3Input = tf.concat([resizedT1, resizedT2, step1Binary, step2Binary], 2).expandDims(0)

          // Predict
          const step3Prediction = unetModel.modelStep3.predict(step3Input) as tf.Tensor4D

          // Get class with highest probability
          const segmentation = tf.argMax(step3Prediction, 3).squeeze() as tf.Tensor2D

          // Resize back to original dimensions
          const resizedPrediction = tf.image.resizeBilinear(segmentation.expandDims(2), [width, height], true).squeeze()

          // Render to canvas with colormap
          renderToCanvas(resizedPrediction, canvasRef.current!, true)

          // Calculate Dice coefficient if ground truth is available
          if (groundTruthVolume) {
            const gtTensor = tf.tensor3d(Array.from(groundTruthVolume.data), groundTruthVolume.header.dimensions)
            const gtSlice = gtTensor.slice([0, 0, sliceIndex], [width, height, 1]).squeeze()

            // Render ground truth to canvas with colormap
            renderToCanvas(gtSlice, gtCanvasRef.current!, true)

            // Calculate multi-class Dice coefficient
            const dice = await calculateMultiClassDice(resizedPrediction, gtSlice)
            setDiceCoefficient(dice)
          }

          // Clean up
          t1Slice.dispose()
          t2Slice.dispose()
          resizedT1.dispose()
          resizedT2.dispose()
          step1Input.dispose()
          step1Prediction.dispose()
          step1Binary.dispose()
          step2Input.dispose()
          step2Prediction.dispose()
          step2Binary.dispose()
          step3Input.dispose()
          step3Prediction.dispose()
          segmentation.dispose()
          resizedPrediction.dispose()
        }

        // Clean up
        t1Tensor.dispose()
        t2Tensor.dispose()
        t1Normalized.dispose()
        t2Normalized.dispose()
      } catch (error) {
        console.error("Error generating prediction:", error)
        setError(`Error generating prediction: ${error}`)
      } finally {
        setIsProcessing(false)
      }
    }

    generatePrediction()
  }, [t1Volume, t2Volume, groundTruthVolume, unetModel, step, sliceIndex])

  // Function to render tensor to canvas
  const renderToCanvas = (tensor: tf.Tensor, canvas: HTMLCanvasElement, isSegmentation: boolean) => {
    if (!canvas) return

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
        // Binary mask visualization
        for (let i = 0; i < height * width; i++) {
          const value = tensorData[i] > 0.5 ? 1 : 0

          if (value === 1) {
            // Foreground - blue
            data[i * 4] = 30
            data[i * 4 + 1] = 144
            data[i * 4 + 2] = 255
            data[i * 4 + 3] = 200
          } else {
            // Background - transparent
            data[i * 4] = 0
            data[i * 4 + 1] = 0
            data[i * 4 + 2] = 0
            data[i * 4 + 3] = 0
          }
        }
      }

      ctx.putImageData(imageData, 0, 0)
    })
  }

  // Calculate Dice coefficient for binary segmentation
  const calculateDice = async (prediction: tf.Tensor, groundTruth: tf.Tensor): Promise<number> => {
    return tf.tidy(() => {
      // Ensure both tensors are binary
      const predBinary = prediction.greater(0.5)
      const gtBinary = groundTruth.greater(0.5)

      // Calculate intersection and union
      const intersection = predBinary.logicalAnd(gtBinary).sum().dataSync()[0]
      const predSum = predBinary.sum().dataSync()[0]
      const gtSum = gtBinary.sum().dataSync()[0]

      // Calculate Dice coefficient
      if (predSum + gtSum === 0) return 0
      return (2 * intersection) / (predSum + gtSum)
    })
  }

  // Calculate multi-class Dice coefficient
  const calculateMultiClassDice = async (prediction: tf.Tensor, groundTruth: tf.Tensor): Promise<number> => {
    return tf.tidy(() => {
      let totalDice = 0
      let classCount = 0

      // Calculate Dice for each class (skip background class 0)
      for (let c = 1; c < 4; c++) {
        // Create binary masks for current class
        const predMask = prediction.equal(tf.scalar(c))
        const gtMask = groundTruth.equal(tf.scalar(c))

        // Calculate intersection and union
        const intersection = predMask.logicalAnd(gtMask).sum().dataSync()[0]
        const predSum = predMask.sum().dataSync()[0]
        const gtSum = gtMask.sum().dataSync()[0]

        // Only include classes that are present in ground truth
        if (gtSum > 0) {
          const dice = (2 * intersection) / (predSum + gtSum)
          totalDice += dice
          classCount++
        }
      }

      // Return average Dice coefficient
      return classCount > 0 ? totalDice / classCount : 0
    })
  }

  // Handle slice change
  const handleSliceChange = (newSlice: number) => {
    if (!t1Volume) return

    const maxSlice = t1Volume.header.dimensions[2] - 1
    if (newSlice >= 0 && newSlice <= maxSlice) {
      setSliceIndex(newSlice)
    }
  }

  return (
    <Card className="mt-4">
      <CardContent className="pt-6">
        <h3 className="text-lg font-medium mb-4">
          Step {step} Prediction Results
          {step === 1 && " (Brain Segmentation)"}
          {step === 2 && " (White Matter Segmentation)"}
          {step === 3 && " (Multi-class Segmentation)"}
        </h3>

        {error && <div className="text-red-500 mb-4">{error}</div>}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="text-sm font-medium mb-2">Prediction</h4>
            <div className="bg-gray-100 rounded-md aspect-square relative">
              {isProcessing ? (
                <div className="absolute inset-0 flex items-center justify-center">
                  <p>Processing...</p>
                </div>
              ) : (
                <canvas ref={canvasRef} className="w-full h-full" />
              )}
            </div>
          </div>

          {groundTruthVolume && (
            <div>
              <h4 className="text-sm font-medium mb-2">Ground Truth</h4>
              <div className="bg-gray-100 rounded-md aspect-square relative">
                <canvas ref={gtCanvasRef} className="w-full h-full" />
              </div>
            </div>
          )}
        </div>

        {t1Volume && (
          <div className="mt-4">
            <h4 className="text-sm font-medium mb-2">Slice: {sliceIndex}</h4>
            <div className="flex items-center gap-2">
              <button
                onClick={() => handleSliceChange(sliceIndex - 1)}
                className="px-2 py-1 bg-gray-200 rounded"
                disabled={sliceIndex <= 0}
              >
                &lt;
              </button>
              <div className="flex-1 h-2 bg-gray-200 rounded-full">
                <div
                  className="h-full bg-blue-500 rounded-full"
                  style={{
                    width: `${(sliceIndex / (t1Volume.header.dimensions[2] - 1)) * 100}%`,
                  }}
                ></div>
              </div>
              <button
                onClick={() => handleSliceChange(sliceIndex + 1)}
                className="px-2 py-1 bg-gray-200 rounded"
                disabled={sliceIndex >= t1Volume.header.dimensions[2] - 1}
              >
                &gt;
              </button>
            </div>
          </div>
        )}

        {diceCoefficient !== null && (
          <div className="mt-4 p-4 bg-gray-50 rounded-md border">
            <h4 className="font-medium mb-2">Dice Coefficient</h4>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-4 bg-gray-200 rounded-full">
                <div
                  className={`h-full rounded-full ${
                    diceCoefficient > 0.8 ? "bg-green-500" : diceCoefficient > 0.6 ? "bg-yellow-500" : "bg-red-500"
                  }`}
                  style={{ width: `${diceCoefficient * 100}%` }}
                ></div>
              </div>
              <span className="font-bold">{(diceCoefficient * 100).toFixed(2)}%</span>
            </div>
            <p className="text-sm mt-2">
              {diceCoefficient > 0.8
                ? "Excellent segmentation quality!"
                : diceCoefficient > 0.6
                  ? "Good segmentation quality."
                  : "Poor segmentation quality. Consider training longer or with more data."}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
