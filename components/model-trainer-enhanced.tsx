"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Slider } from "@/components/ui/slider"
import { EnhancedUNet } from "@/lib/unet-enhanced"
import type { AnalyzeVolume } from "@/lib/analyze-parser"
import * as tf from "@tensorflow/tfjs" // Import tf directly to ensure it's available
import { Checkbox } from "@/components/ui/checkbox"
import StepPredictionViewer from "./step-prediction-viewer"

interface ModelTrainerEnhancedProps {
  onModelTrained: () => void
  trainingVolumes: { t1: AnalyzeVolume; t2: AnalyzeVolume; groundTruth?: AnalyzeVolume }[]
  validationVolumes: { t1: AnalyzeVolume; t2: AnalyzeVolume; groundTruth?: AnalyzeVolume }[]
}

export default function ModelTrainerEnhanced({
                                               onModelTrained,
                                               trainingVolumes,
                                               validationVolumes,
                                             }: ModelTrainerEnhancedProps) {
  const [isTraining, setIsTraining] = useState(false)
  const [progress, setProgress] = useState(0)
  const [epoch, setEpoch] = useState(0)
  const [totalEpochs, setTotalEpochs] = useState(100)
  const [batchSize, setBatchSize] = useState(1) // Set to 1 as in the Python code
  const [learningRate, setLearningRate] = useState(0.0001)
  const [trainingLogs, setTrainingLogs] = useState<string[]>([])
  const [metrics, setMetrics] = useState<{ [key: string]: number[] }>({
    loss: [],
    accuracy: [],
    val_loss: [],
    val_accuracy: [],
    dice_coef: [],
  })
  const [tfReady, setTfReady] = useState(false)
  const [modelTrained, setModelTrained] = useState(false)
  const [currentStep, setCurrentStep] = useState(1) // Track which step we're on (1, 2, or 3)
  const [patience, setPatience] = useState(10) // Early stopping patience
  const [saveInterval, setSaveInterval] = useState(10) // Save model every N epochs
  const [stopOnSave, setStopOnSave] = useState(true) // Stop training when model is saved

  // Add state for prediction results
  const [predictionResults, setPredictionResults] = useState<{
    step: number
    image: string | null
    dice: number | null
  } | null>(null)

  // Add a state to track validation subjects
  const [testingSubjects, setTestingSubjects] = useState<number>(0)
  const [validationSubjects, setValidationSubjects] = useState<number>(0)

  const [stepCompleted, setStepCompleted] = useState<boolean>(false)
  const [modelSaved, setModelSaved] = useState<boolean>(false)

  const unetRef = useRef<EnhancedUNet | null>(null)
  const shouldStopRef = useRef<boolean>(false) // Reference to track if we should stop training

  // Initialize TensorFlow.js first
  useEffect(() => {
    const initTf = async () => {
      try {
        // Ensure TensorFlow.js is initialized
        await tf.ready()
        console.log("TensorFlow.js initialized successfully")
        setTfReady(true)
      } catch (error) {
        console.error("Failed to initialize TensorFlow.js:", error)
        setTrainingLogs((prev) => [...prev, `Failed to initialize TensorFlow.js: ${error}`])
      }
    }

    initTf()
  }, [])

  // Initialize the model after TensorFlow.js is ready
  useEffect(() => {
    if (!tfReady) return

    const initModel = async () => {
      try {
        unetRef.current = new EnhancedUNet()
        await unetRef.current.initialize()
        setTrainingLogs((prev) => [...prev, "Model initialized successfully"])

        // Check if a model is already saved
        try {
          const modelInfo = await tf.io.listModels()
          if (modelInfo["indexeddb://unet-iseg-model-step1"]) {
            setModelTrained(true)
            setTrainingLogs((prev) => [...prev, "Found previously trained model in storage"])
          }
        } catch (e) {
          console.log("No previously trained model found")
        }
      } catch (error) {
        console.error("Failed to initialize model:", error)
        setTrainingLogs((prev) => [...prev, `Failed to initialize model: ${error}`])
      }
    }

    initModel()
  }, [tfReady])

  // Update the useEffect that initializes the model to also count validation subjects
  useEffect(() => {
    // Count testing and validation subjects
    if (validationVolumes.length >= 4) {
      setTestingSubjects(2)
      setValidationSubjects(2)
    } else if (validationVolumes.length >= 2) {
      setTestingSubjects(validationVolumes.length - 2)
      setValidationSubjects(2)
    } else {
      setTestingSubjects(0)
      setValidationSubjects(validationVolumes.length)
    }
  }, [validationVolumes])

  // Function to filter slices with significant brain tissue (>100 pixels with values > 0)
  const filterSlices = (volume: AnalyzeVolume): number[] => {
    const { dimensions } = volume.header
    const [width, height, depth] = dimensions
    const validSlices: number[] = []

    for (let z = 0; z < depth; z++) {
      let nonZeroPixels = 0

      // Count non-zero pixels in this slice
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const volumeIndex = x + y * width + z * width * height
          if (volume.data[volumeIndex] > 0) {
            nonZeroPixels++
          }
        }
      }

      // If slice has more than 100 non-zero pixels, consider it valid
      if (nonZeroPixels > 100) {
        validSlices.push(z)
      }
    }

    return validSlices
  }

  // Function to prepare data for Step 1 (similar to preparation_data in Python)
  const prepareDataStep1 = async (
      volumes: { t1: AnalyzeVolume; t2: AnalyzeVolume; groundTruth?: AnalyzeVolume }[],
  ): Promise<{ xs: tf.Tensor4D; ys: tf.Tensor4D }> => {
    setTrainingLogs((prev) => [...prev, "Preparing data for Step 1..."])

    // Count total valid slices
    let totalValidSlices = 0
    const validSlicesPerVolume: number[][] = []

    for (const volume of volumes) {
      if (!volume.groundTruth) continue

      const validSlices = filterSlices(volume.groundTruth)
      validSlicesPerVolume.push(validSlices)
      totalValidSlices += validSlices.length
    }

    setTrainingLogs((prev) => [
      ...prev,
      `Found ${totalValidSlices} valid slices across ${validSlicesPerVolume.length} volumes`,
    ])

    // Create tensors to hold all valid slices
    const inputTensors: tf.Tensor4D[] = []
    const targetTensors: tf.Tensor4D[] = []

    // Process each volume
    for (let i = 0; i < volumes.length; i++) {
      const volume = volumes[i]
      if (!volume.groundTruth) continue

      const validSlices = validSlicesPerVolume[i]

      for (const z of validSlices) {
        // Extract T1 slice
        const t1Slice = extractSlice(volume.t1, z)
        // Extract T2 slice
        const t2Slice = extractSlice(volume.t2, z)
        // Extract ground truth slice
        const gtSlice = extractSlice(volume.groundTruth, z)

        // Resize to 128x128
        const resizedT1 = tf.tidy(() => {
          const t1Tensor = tf.tensor(t1Slice)
          const reshapedT1 = t1Tensor.reshape([t1Slice.length, t1Slice[0].length, 1])
          return tf.image.resizeBilinear(reshapedT1, [128, 128])
        })

        const resizedT2 = tf.tidy(() => {
          const t2Tensor = tf.tensor(t2Slice)
          const reshapedT2 = t2Tensor.reshape([t2Slice.length, t2Slice[0].length, 1])
          return tf.image.resizeBilinear(reshapedT2, [128, 128])
        })

        // Combine T1 and T2 along the channel dimension
        const combined = tf.tidy(() => {
          return tf.concat([resizedT1, resizedT2], 2).expandDims(0)
        })

        // For Step 1: Binary segmentation (brain vs. background)
        const binaryGt = tf.tidy(() => {
          const gtTensor = tf.tensor(gtSlice)
          const reshapedGt = gtTensor.reshape([gtSlice.length, gtSlice[0].length, 1])
          const resizedGt = tf.image.resizeBilinear(reshapedGt, [128, 128])
          // Create binary mask: 1 where GT > 0, 0 elsewhere
          const binaryMask = resizedGt.greater(0).toFloat()
          // For Step 1, we need a binary output (1 channel)
          return binaryMask.expandDims(0)
        })

        inputTensors.push(combined as tf.Tensor4D)
        targetTensors.push(binaryGt as tf.Tensor4D)
      }
    }

    // Concatenate all slices
    const xs = tf.concat(inputTensors, 0)
    const ys = tf.concat(targetTensors, 0)

    setTrainingLogs((prev) => [...prev, `Prepared data shapes: xs=${xs.shape}, ys=${ys.shape}`])

    // Clean up individual tensors
    inputTensors.forEach((t) => t.dispose())
    targetTensors.forEach((t) => t.dispose())

    return { xs, ys }
  }

  // Function to prepare data for Step 2 (similar to preparing_data_step2 in Python)
  const prepareDataStep2 = async (
      volumes: { t1: AnalyzeVolume; t2: AnalyzeVolume; groundTruth?: AnalyzeVolume }[],
  ): Promise<{ xs: tf.Tensor4D; ys: tf.Tensor4D }> => {
    setTrainingLogs((prev) => [...prev, "Preparing data for Step 2..."])

    if (!unetRef.current || !unetRef.current.modelStep1) {
      throw new Error("Step 1 model not initialized")
    }

    // Count total valid slices
    let totalValidSlices = 0
    const validSlicesPerVolume: number[][] = []

    for (const volume of volumes) {
      if (!volume.groundTruth) continue

      const validSlices = filterSlices(volume.groundTruth)
      validSlicesPerVolume.push(validSlices)
      totalValidSlices += validSlices.length
    }

    setTrainingLogs((prev) => [
      ...prev,
      `Found ${totalValidSlices} valid slices across ${validSlicesPerVolume.length} volumes`,
    ])

    // Create tensors to hold all valid slices
    const inputTensors: tf.Tensor4D[] = []
    const targetTensors: tf.Tensor4D[] = []

    // Process each volume
    for (let i = 0; i < volumes.length; i++) {
      const volume = volumes[i]
      if (!volume.groundTruth) continue

      const validSlices = validSlicesPerVolume[i]

      for (const z of validSlices) {
        // Extract T1 slice
        const t1Slice = extractSlice(volume.t1, z)
        // Extract T2 slice
        const t2Slice = extractSlice(volume.t2, z)
        // Extract ground truth slice
        const gtSlice = extractSlice(volume.groundTruth, z)

        // Resize to 128x128
        const resizedT1 = tf.tidy(() => {
          const t1Tensor = tf.tensor(t1Slice)
          const reshapedT1 = t1Tensor.reshape([t1Slice.length, t1Slice[0].length, 1])
          return tf.image.resizeBilinear(reshapedT1, [128, 128])
        })

        const resizedT2 = tf.tidy(() => {
          const t2Tensor = tf.tensor(t2Slice)
          const reshapedT2 = t2Tensor.reshape([t2Slice.length, t2Slice[0].length, 1])
          return tf.image.resizeBilinear(reshapedT2, [128, 128])
        })

        // Get Step 1 prediction
        const step1Input = tf.tidy(() => {
          return tf.concat([resizedT1, resizedT2], 2).expandDims(0)
        })

        const step1Prediction = tf.tidy(() => {
          const pred = unetRef.current!.modelStep1!.predict(step1Input) as tf.Tensor4D
          // Convert to binary mask (threshold at 0.5)
          return pred.greater(0.5).toFloat().squeeze(0)
        })

        // Combine T1, T2, and Step 1 prediction
        const combined = tf.tidy(() => {
          return tf.concat([resizedT1, resizedT2, step1Prediction], 2).expandDims(0)
        })

        // For Step 2: White matter segmentation (binary)
        const whiteGt = tf.tidy(() => {
          const gtTensor = tf.tensor(gtSlice)
          const reshapedGt = gtTensor.reshape([gtSlice.length, gtSlice[0].length, 1])
          const resizedGt = tf.image.resizeBilinear(reshapedGt, [128, 128])
          // Create binary mask: 1 where GT > 10 (white matter), 0 elsewhere
          const binaryMask = resizedGt.greater(10).toFloat()
          // For Step 2, we need a binary output (1 channel)
          return binaryMask.expandDims(0)
        })

        inputTensors.push(combined as tf.Tensor4D)
        targetTensors.push(whiteGt as tf.Tensor4D)

        // Clean up
        step1Input.dispose()
        step1Prediction.dispose()
      }
    }

    // Concatenate all slices
    const xs = tf.concat(inputTensors, 0)
    const ys = tf.concat(targetTensors, 0)

    setTrainingLogs((prev) => [...prev, `Prepared data shapes: xs=${xs.shape}, ys=${ys.shape}`])

    // Clean up individual tensors
    inputTensors.forEach((t) => t.dispose())
    targetTensors.forEach((t) => t.dispose())

    return { xs, ys }
  }

  // Function to prepare data for Step 3
  const prepareDataStep3 = async (
      volumes: { t1: AnalyzeVolume; t2: AnalyzeVolume; groundTruth?: AnalyzeVolume }[],
  ): Promise<{ xs: tf.Tensor4D; ys: tf.Tensor4D }> => {
    setTrainingLogs((prev) => [...prev, "Preparing data for Step 3..."])

    if (!unetRef.current || !unetRef.current.modelStep1 || !unetRef.current.modelStep2) {
      throw new Error("Step 1 or Step 2 model not initialized")
    }

    // Count total valid slices
    let totalValidSlices = 0
    const validSlicesPerVolume: number[][] = []

    for (const volume of volumes) {
      if (!volume.groundTruth) continue

      const validSlices = filterSlices(volume.groundTruth)
      validSlicesPerVolume.push(validSlices)
      totalValidSlices += validSlices.length
    }

    setTrainingLogs((prev) => [
      ...prev,
      `Found ${totalValidSlices} valid slices across ${validSlicesPerVolume.length} volumes`,
    ])

    // Create tensors to hold all valid slices
    const inputTensors: tf.Tensor4D[] = []
    const targetTensors: tf.Tensor4D[] = []

    // Process each volume
    for (let i = 0; i < volumes.length; i++) {
      const volume = volumes[i]
      if (!volume.groundTruth) continue

      const validSlices = validSlicesPerVolume[i]

      for (const z of validSlices) {
        // Extract T1 slice
        const t1Slice = extractSlice(volume.t1, z)
        // Extract T2 slice
        const t2Slice = extractSlice(volume.t2, z)
        // Extract ground truth slice
        const gtSlice = extractSlice(volume.groundTruth, z)

        // Resize to 128x128
        const resizedT1 = tf.tidy(() => {
          const t1Tensor = tf.tensor(t1Slice)
          const reshapedT1 = t1Tensor.reshape([t1Slice.length, t1Slice[0].length, 1])
          return tf.image.resizeBilinear(reshapedT1, [128, 128])
        })

        const resizedT2 = tf.tidy(() => {
          const t2Tensor = tf.tensor(t2Slice)
          const reshapedT2 = t2Tensor.reshape([t2Slice.length, t2Slice[0].length, 1])
          return tf.image.resizeBilinear(reshapedT2, [128, 128])
        })

        // Get Step 1 prediction
        const step1Input = tf.tidy(() => {
          return tf.concat([resizedT1, resizedT2], 2).expandDims(0)
        })

        const step1Prediction = tf.tidy(() => {
          const pred = unetRef.current!.modelStep1!.predict(step1Input) as tf.Tensor4D
          return pred.greater(0.5).toFloat().squeeze(0)
        })

        // Get Step 2 prediction
        const step2Input = tf.tidy(() => {
          return tf.concat([resizedT1, resizedT2, step1Prediction], 2).expandDims(0)
        })

        const step2Prediction = tf.tidy(() => {
          const pred = unetRef.current!.modelStep2!.predict(step2Input) as tf.Tensor4D
          return pred.greater(0.5).toFloat().squeeze(0)
        })

        // Combine T1, T2, Step 1 and Step 2 predictions
        const combined = tf.tidy(() => {
          return tf.concat([resizedT1, resizedT2, step1Prediction, step2Prediction], 2).expandDims(0)
        })

        // For Step 3: Multi-class segmentation
        const multiClassGt = tf.tidy(() => {
          const gtTensor = tf.tensor(gtSlice)
          const reshapedGt = gtTensor.reshape([gtSlice.length, gtSlice[0].length, 1])
          const resizedGt = tf.image.resizeBilinear(reshapedGt, [128, 128])
          // Scale to [0,3] range and round to get class indices
          const gtClasses = tf.cast(tf.round(resizedGt.mul(3)), "int32").reshape([128, 128])
          // One-hot encode with 4 classes
          return tf.oneHot(gtClasses, 4).expandDims(0)
        })

        inputTensors.push(combined as tf.Tensor4D)
        targetTensors.push(multiClassGt as tf.Tensor4D)

        // Clean up
        step1Input.dispose()
        step1Prediction.dispose()
        step2Input.dispose()
        step2Prediction.dispose()
      }
    }

    // Concatenate all slices
    const xs = tf.concat(inputTensors, 0)
    const ys = tf.concat(targetTensors, 0)

    setTrainingLogs((prev) => [...prev, `Prepared data shapes: xs=${xs.shape}, ys=${ys.shape}`])

    // Clean up individual tensors
    inputTensors.forEach((t) => t.dispose())
    targetTensors.forEach((t) => t.dispose())

    return { xs, ys }
  }

  // Function to make a prediction and calculate Dice coefficient after a step is completed
  const evaluateCurrentStep = async (validationVolume: {
    t1: AnalyzeVolume
    t2: AnalyzeVolume
    groundTruth?: AnalyzeVolume
  }) => {
    if (!unetRef.current || !validationVolume.groundTruth) return

    try {
      setTrainingLogs((prev) => [...prev, `Evaluating Step ${currentStep} model...`])

      // Get a middle slice for visualization
      const { dimensions } = validationVolume.t1.header
      const middleSlice = Math.floor(dimensions[2] / 2)

      // Extract slices
      const t1Slice = extractSlice(validationVolume.t1, middleSlice)
      const t2Slice = extractSlice(validationVolume.t2, middleSlice)
      const gtSlice = extractSlice(validationVolume.groundTruth, middleSlice)

      // Convert to tensors
      const t1Tensor = tf.tensor(t1Slice)
      const t2Tensor = tf.tensor(t2Slice)
      const gtTensor = tf.tensor(gtSlice)

      // Reshape and resize
      const reshapedT1 = t1Tensor.reshape([t1Slice.length, t1Slice[0].length, 1])
      const reshapedT2 = t2Tensor.reshape([t2Slice.length, t2Slice[0].length, 1])
      const reshapedGt = gtTensor.reshape([gtSlice.length, gtSlice[0].length, 1])

      const resizedT1 = tf.image.resizeBilinear(reshapedT1, [128, 128])
      const resizedT2 = tf.image.resizeBilinear(reshapedT2, [128, 128])
      const resizedGt = tf.image.resizeBilinear(reshapedGt, [128, 128])

      let prediction: tf.Tensor

      if (currentStep === 1) {
        // Step 1: Brain segmentation
        const input = tf.concat([resizedT1, resizedT2], 2).expandDims(0)
        const output = unetRef.current.modelStep1!.predict(input) as tf.Tensor4D
        prediction = output.greater(0.5).toFloat().squeeze()

        // Calculate Dice for brain segmentation
        const gtBinary = resizedGt.greater(0).toFloat()
        const dice = await unetRef.current.dice(prediction, gtBinary)

        setTrainingLogs((prev) => [...prev, `Step 1 Dice coefficient: ${dice.toFixed(4)}`])
        setMetrics((prev) => ({ ...prev, dice_coef: [...(prev.dice_coef || []), dice] }))

        // Convert prediction to image for display
        const canvas = document.createElement("canvas")
        canvas.width = 128
        canvas.height = 128
        const ctx = canvas.getContext("2d")

        if (ctx) {
          const imageData = ctx.createImageData(128, 128)
          const predData = prediction.dataSync()

          for (let i = 0; i < predData.length; i++) {
            const value = predData[i] > 0.5 ? 255 : 0
            imageData.data[i * 4] = value // R
            imageData.data[i * 4 + 1] = value // G
            imageData.data[i * 4 + 2] = value // B
            imageData.data[i * 4 + 3] = 255 // A
          }

          ctx.putImageData(imageData, 0, 0)
          const dataUrl = canvas.toDataURL()

          setPredictionResults({
            step: 1,
            image: dataUrl,
            dice: dice,
          })
        }

        // Clean up
        input.dispose()
        output.dispose()
        gtBinary.dispose()
      } else if (currentStep === 2) {
        // Step 1 prediction
        const step1Input = tf.concat([resizedT1, resizedT2], 2).expandDims(0)
        const step1Output = unetRef.current.modelStep1!.predict(step1Input) as tf.Tensor4D
        const step1Pred = step1Output.greater(0.5).toFloat().squeeze()

        // Step 2: White matter segmentation
        const step2Input = tf.concat([resizedT1, resizedT2, step1Pred.expandDims(2)], 2).expandDims(0)
        const step2Output = unetRef.current.modelStep2!.predict(step2Input) as tf.Tensor4D
        prediction = step2Output.greater(0.5).toFloat().squeeze()

        // Calculate Dice for white matter segmentation
        const gtWhite = resizedGt.greater(10).toFloat()
        const dice = await unetRef.current.dice(prediction, gtWhite)

        setTrainingLogs((prev) => [...prev, `Step 2 Dice coefficient: ${dice.toFixed(4)}`])
        setMetrics((prev) => ({ ...prev, dice_coef: [...(prev.dice_coef || []), dice] }))

        // Convert prediction to image for display
        const canvas = document.createElement("canvas")
        canvas.width = 128
        canvas.height = 128
        const ctx = canvas.getContext("2d")

        if (ctx) {
          const imageData = ctx.createImageData(128, 128)
          const predData = prediction.dataSync()

          for (let i = 0; i < predData.length; i++) {
            // White matter in orange
            const value = predData[i] > 0.5
            imageData.data[i * 4] = value ? 255 : 0 // R
            imageData.data[i * 4 + 1] = value ? 165 : 0 // G
            imageData.data[i * 4 + 2] = value ? 0 : 0 // B
            imageData.data[i * 4 + 3] = value ? 200 : 0 // A
          }

          ctx.putImageData(imageData, 0, 0)
          const dataUrl = canvas.toDataURL()

          setPredictionResults({
            step: 2,
            image: dataUrl,
            dice: dice,
          })
        }

        // Clean up
        step1Input.dispose()
        step1Output.dispose()
        step1Pred.dispose()
        step2Input.dispose()
        step2Output.dispose()
        gtWhite.dispose()
      } else if (currentStep === 3) {
        // Step 1 prediction
        const step1Input = tf.concat([resizedT1, resizedT2], 2).expandDims(0)
        const step1Output = unetRef.current.modelStep1!.predict(step1Input) as tf.Tensor4D
        const step1Pred = step1Output.greater(0.5).toFloat().squeeze()

        // Step 2 prediction
        const step2Input = tf.concat([resizedT1, resizedT2, step1Pred.expandDims(2)], 2).expandDims(0)
        const step2Output = unetRef.current.modelStep2!.predict(step2Input) as tf.Tensor4D
        const step2Pred = step2Output.greater(0.5).toFloat().squeeze()

        // Step 3: Multi-class segmentation
        const step3Input = tf
            .concat([resizedT1, resizedT2, step1Pred.expandDims(2), step2Pred.expandDims(2)], 2)
            .expandDims(0)

        const step3Output = unetRef.current.modelStep3!.predict(step3Input) as tf.Tensor4D
        const segmentation = tf.argMax(step3Output, 3).squeeze()

        // Calculate average Dice across classes
        const gtClasses = tf.cast(tf.round(resizedGt.mul(3)), "int32").squeeze()

        let totalDice = 0
        let classCount = 0

        // Calculate Dice for each class (1, 2, 3)
        for (let c = 1; c < 4; c++) {
          const predMask = segmentation.equal(tf.scalar(c, "int32"))
          const gtMask = gtClasses.equal(tf.scalar(c, "int32"))

          const dice = await unetRef.current.dice(predMask, gtMask)
          if (dice >= 0) {
            // Only count classes that are present
            totalDice += dice
            classCount++
          }

          predMask.dispose()
          gtMask.dispose()
        }

        const avgDice = classCount > 0 ? totalDice / classCount : 0

        setTrainingLogs((prev) => [...prev, `Step 3 Average Dice coefficient: ${avgDice.toFixed(4)}`])
        setMetrics((prev) => ({ ...prev, dice_coef: [...(prev.dice_coef || []), avgDice] }))

        // Convert prediction to image for display
        const canvas = document.createElement("canvas")
        canvas.width = 128
        canvas.height = 128
        const ctx = canvas.getContext("2d")

        if (ctx) {
          const imageData = ctx.createImageData(128, 128)
          const predData = segmentation.dataSync()

          for (let i = 0; i < predData.length; i++) {
            // Multi-class colormap
            const classValue = predData[i]

            if (classValue === 0) {
              // Background
              imageData.data[i * 4] = 0
              imageData.data[i * 4 + 1] = 0
              imageData.data[i * 4 + 2] = 0
              imageData.data[i * 4 + 3] = 0
            } else if (classValue === 1) {
              // CSF
              imageData.data[i * 4] = 65
              imageData.data[i * 4 + 1] = 105
              imageData.data[i * 4 + 2] = 225
              imageData.data[i * 4 + 3] = 200
            } else if (classValue === 2) {
              // GM
              imageData.data[i * 4] = 50
              imageData.data[i * 4 + 1] = 205
              imageData.data[i * 4 + 2] = 50
              imageData.data[i * 4 + 3] = 200
            } else if (classValue === 3) {
              // WM
              imageData.data[i * 4] = 255
              imageData.data[i * 4 + 1] = 165
              imageData.data[i * 4 + 2] = 0
              imageData.data[i * 4 + 3] = 200
            }
          }

          ctx.putImageData(imageData, 0, 0)
          const dataUrl = canvas.toDataURL()

          setPredictionResults({
            step: 3,
            image: dataUrl,
            dice: avgDice,
          })
        }

        // Clean up
        step1Input.dispose()
        step1Output.dispose()
        step1Pred.dispose()
        step2Input.dispose()
        step2Output.dispose()
        step2Pred.dispose()
        step3Input.dispose()
        step3Output.dispose()
        segmentation.dispose()
        gtClasses.dispose()
      }

      // Clean up
      t1Tensor.dispose()
      t2Tensor.dispose()
      gtTensor.dispose()
      reshapedT1.dispose()
      reshapedT2.dispose()
      reshapedGt.dispose()
      resizedT1.dispose()
      resizedT2.dispose()
      resizedGt.dispose()
    } catch (error) {
      console.error("Error during evaluation:", error)
      setTrainingLogs((prev) => [...prev, `Error during evaluation: ${error}`])
    }
  }

  // Implement early stopping with model saving
  const createEarlyStopping = (patience = 10) => {
    let bestValLoss = Number.POSITIVE_INFINITY
    let bestWeights: tf.NamedTensorMap | null = null
    let waitCount = 0

    return {
      onEpochEnd: async (epoch: number, logs: any) => {
        const currentValLoss = logs.val_loss

        if (currentValLoss < bestValLoss) {
          bestValLoss = currentValLoss
          waitCount = 0

          // Save best weights
          if (unetRef.current) {
            if (currentStep === 1 && unetRef.current.modelStep1) {
              bestWeights = await tf.tidy(() => {
                return unetRef.current!.modelStep1!.getWeights().reduce((result, w, i) => {
                  result[`weight_${i}`] = w.clone()
                  return result
                }, {} as tf.NamedTensorMap)
              })
            } else if (currentStep === 2 && unetRef.current.modelStep2) {
              bestWeights = await tf.tidy(() => {
                return unetRef.current!.modelStep2!.getWeights().reduce((result, w, i) => {
                  result[`weight_${i}`] = w.clone()
                  return result
                }, {} as tf.NamedTensorMap)
              })
            } else if (currentStep === 3 && unetRef.current.modelStep3) {
              bestWeights = await tf.tidy(() => {
                return unetRef.current!.modelStep3!.getWeights().reduce((result, w, i) => {
                  result[`weight_${i}`] = w.clone()
                  return result
                }, {} as tf.NamedTensorMap)
              })
            }
          }
        } else {
          waitCount++
        }

        // Save model at specified intervals
        if ((epoch + 1) % saveInterval === 0) {
          try {
            setTrainingLogs((prev) => [...prev, `Saving model at epoch ${epoch + 1}...`])

            if (currentStep === 1) {
              await unetRef.current?.saveModel(unetRef.current.modelPathStep1)
            } else if (currentStep === 2) {
              await unetRef.current?.saveModel(unetRef.current.modelPathStep2)
            } else if (currentStep === 3) {
              await unetRef.current?.saveModel(unetRef.current.modelPathStep3)
            }

            setTrainingLogs((prev) => [...prev, `Model saved at epoch ${epoch + 1}`])

            // Check if we should stop after saving
            if (stopOnSave) {
              setTrainingLogs((prev) => [...prev, `Stopping training after saving at epoch ${epoch + 1}`])
              shouldStopRef.current = true
              return true // Signal to stop training
            }
          } catch (error) {
            console.error("Failed to save model:", error)
            setTrainingLogs((prev) => [...prev, `Failed to save model: ${error}`])
          }
        }

        // Check if we should stop due to early stopping
        if (waitCount >= patience) {
          setTrainingLogs((prev) => [...prev, `Early stopping triggered after ${epoch + 1} epochs`])

          // Restore best weights
          if (bestWeights && unetRef.current) {
            const weights = Object.values(bestWeights).map((w) => w.clone())

            if (currentStep === 1 && unetRef.current.modelStep1) {
              unetRef.current.modelStep1.setWeights(weights)
            } else if (currentStep === 2 && unetRef.current.modelStep2) {
              unetRef.current.modelStep2.setWeights(weights)
            } else if (currentStep === 3 && unetRef.current.modelStep3) {
              unetRef.current.modelStep3.setWeights(weights)
            }

            setTrainingLogs((prev) => [...prev, "Restored best weights"])
          }

          return true // Signal to stop training
        }

        // Check if we should stop due to external signal
        if (shouldStopRef.current) {
          setTrainingLogs((prev) => [...prev, `Training stopped at epoch ${epoch + 1}`])
          return true
        }

        return false
      },

      cleanup: () => {
        // Clean up tensors
        if (bestWeights) {
          Object.values(bestWeights).forEach((w) => w.dispose())
          bestWeights = null
        }
      },
    }
  }

  const startTraining = async () => {
    if (!tfReady) {
      setTrainingLogs((prev) => [...prev, "TensorFlow.js not initialized yet. Please wait."])
      return
    }

    if (!unetRef.current) {
      setTrainingLogs((prev) => [...prev, "Model not initialized"])
      return
    }

    if (trainingVolumes.length === 0) {
      setTrainingLogs((prev) => [...prev, "No training data available"])
      return
    }

    setIsTraining(true)
    setProgress(0)
    setEpoch(0)
    shouldStopRef.current = false
    setPredictionResults(null)
    setTrainingLogs((prev) => [...prev, `Starting training for Step ${currentStep}...`])

    try {
      // Create callbacks with early stopping
      const earlyStopping = createEarlyStopping(patience)

      const callbacks = [
        {
          onEpochBegin: async (epoch: number) => {
            setEpoch(epoch)
            setTrainingLogs((prev) => [...prev, `Starting epoch ${epoch + 1}/${totalEpochs}`])
          },
          onEpochEnd: async (epoch: number, logs: any) => {
            setEpoch(epoch + 1)
            setProgress(((epoch + 1) / totalEpochs) * 100)

            // Update metrics
            setMetrics((prev) => {
              const newMetrics = { ...prev }
              for (const key in logs) {
                if (!newMetrics[key]) newMetrics[key] = []
                newMetrics[key].push(logs[key])
              }
              return newMetrics
            })

            setTrainingLogs((prev) => [
              ...prev,
              `Epoch ${epoch + 1}/${totalEpochs} - loss: ${logs.loss?.toFixed(4) || "N/A"} - accuracy: ${logs.accuracy?.toFixed(4) || "N/A"} - val_loss: ${logs.val_loss?.toFixed(4) || "N/A"}`,
            ])

            // Save model every 10 epochs
            if ((epoch + 1) % 10 === 0) {
              try {
                if (currentStep === 1) {
                  await unetRef.current?.saveModel(unetRef.current.modelPathStep1)
                } else if (currentStep === 2) {
                  await unetRef.current?.saveModel(unetRef.current.modelPathStep2)
                } else if (currentStep === 3) {
                  await unetRef.current?.saveModel(unetRef.current.modelPathStep3)
                }
                setTrainingLogs((prev) => [...prev, `Model saved at epoch ${epoch + 1}`])
                setModelSaved(true)

                // Stop training after saving the model
                return true
              } catch (error) {
                console.error("Failed to save model:", error)
                setTrainingLogs((prev) => [...prev, `Failed to save model: ${error}`])
              }
            }

            // Check early stopping
            const shouldStop = await earlyStopping.onEpochEnd(epoch, logs)
            if (shouldStop) {
              setTrainingLogs((prev) => [...prev, "Early stopping triggered"])
              return true // Signal to stop training
            }

            return false
          },
          onBatchEnd: async (batch: number, logs: any) => {
            // Update progress within epoch
            const totalBatches = Math.ceil(trainingVolumes.length / batchSize)
            const batchProgress = (batch / totalBatches) * (100 / totalEpochs)
            const epochProgress = (epoch / totalEpochs) * 100
            setProgress(epochProgress + batchProgress)
          },
        },
      ]

      // Prepare data based on current step
      let trainData: { xs: tf.Tensor4D; ys: tf.Tensor4D }
      let valData: { xs: tf.Tensor4D; ys: tf.Tensor4D }

      if (currentStep === 1) {
        trainData = await prepareDataStep1(trainingVolumes)
        valData = await prepareDataStep1(validationVolumes.length > 0 ? validationVolumes : trainingVolumes.slice(0, 1))

        // Train Step 1 model (binary segmentation)
        setTrainingLogs((prev) => [...prev, "Training Step 1 model (brain segmentation)..."])
        await unetRef.current.trainStep1(
            trainData.xs,
            trainData.ys,
            valData.xs,
            valData.ys,
            totalEpochs,
            batchSize,
            callbacks as any,
        )

        // Save Step 1 model
        await unetRef.current.saveModel(unetRef.current.modelPathStep1)
        setTrainingLogs((prev) => [...prev, "Step 1 model saved"])
      } else if (currentStep === 2) {
        trainData = await prepareDataStep2(trainingVolumes)
        valData = await prepareDataStep2(validationVolumes.length > 0 ? validationVolumes : trainingVolumes.slice(0, 1))

        // Train Step 2 model (white matter segmentation)
        setTrainingLogs((prev) => [...prev, "Training Step 2 model (white matter segmentation)..."])
        await unetRef.current.trainStep2(
            trainData.xs,
            trainData.ys,
            valData.xs,
            valData.ys,
            totalEpochs,
            batchSize,
            callbacks as any,
        )

        // Save Step 2 model
        await unetRef.current.saveModel(unetRef.current.modelPathStep2)
        setTrainingLogs((prev) => [...prev, "Step 2 model saved"])
      } else if (currentStep === 3) {
        trainData = await prepareDataStep3(trainingVolumes)
        valData = await prepareDataStep3(validationVolumes.length > 0 ? validationVolumes : trainingVolumes.slice(0, 1))

        // Train Step 3 model (multi-class segmentation)
        setTrainingLogs((prev) => [...prev, "Training Step 3 model (multi-class segmentation)..."])
        await unetRef.current.trainStep3(
            trainData.xs,
            trainData.ys,
            valData.xs,
            valData.ys,
            totalEpochs,
            batchSize,
            callbacks as any,
        )

        // Save Step 3 model
        await unetRef.current.saveModel(unetRef.current.modelPathStep3)
        setTrainingLogs((prev) => [...prev, "Step 3 model saved"])
      }

      // Clean up tensors
      trainData.xs.dispose()
      trainData.ys.dispose()
      valData.xs.dispose()
      valData.ys.dispose()

      // Clean up early stopping resources
      earlyStopping.cleanup()

      setTrainingLogs((prev) => [...prev, `Step ${currentStep} training complete!`])
      setModelTrained(true)
      setStepCompleted(true)

      // Evaluate the model on a validation volume
      if (validationVolumes.length > 0) {
        await evaluateCurrentStep(validationVolumes[0])
      } else if (trainingVolumes.length > 0) {
        await evaluateCurrentStep(trainingVolumes[0])
      }

      // Move to next step if not at step 3 yet
      if (currentStep < 3) {
        setTrainingLogs((prev) => [...prev, `Ready for Step ${currentStep + 1}`])
        // Don't automatically advance to next step, wait for user to view results
      } else {
        // All steps complete
        onModelTrained()
      }
    } catch (error) {
      console.error(`Step ${currentStep} training failed:`, error)
      setTrainingLogs((prev) => [...prev, `Step ${currentStep} training failed: ${error}`])
    } finally {
      setIsTraining(false)
    }
  }

  const advanceToNextStep = () => {
    if (currentStep < 3) {
      setCurrentStep(currentStep + 1)
      setStepCompleted(false)
      setModelSaved(false)
    } else {
      onModelTrained()
    }
  }

  // Add a function to download the trained model
  const downloadModel = async () => {
    if (!unetRef.current) {
      setTrainingLogs((prev) => [...prev, "Model not initialized"])
      return
    }

    try {
      setTrainingLogs((prev) => [...prev, "Preparing model for download..."])

      if (currentStep === 1) {
        await unetRef.current.saveModel("downloads://unet-iseg-model-step1")
      } else if (currentStep === 2) {
        await unetRef.current.saveModel("downloads://unet-iseg-model-step2")
      } else {
        await unetRef.current.saveModel("downloads://unet-iseg-model-step3")
      }

      setTrainingLogs((prev) => [...prev, `Step ${currentStep} model download initiated`])
    } catch (error) {
      console.error("Failed to download model:", error)
      setTrainingLogs((prev) => [...prev, `Failed to download model: ${error}`])
    }
  }

  // Function to manually stop training
  const stopTraining = () => {
    shouldStopRef.current = true
    setTrainingLogs((prev) => [...prev, "Stopping training..."])
  }

  // Helper function to extract a 2D slice from a volume
  function extractSlice(volume: AnalyzeVolume, sliceIndex: number): number[][] {
    const { dimensions } = volume.header
    const [width, height] = dimensions

    // Create a 2D array for the slice
    const slice: number[][] = Array(height)
        .fill(0)
        .map(() => Array(width).fill(0))

    // Extract the slice data
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const volumeIndex = x + y * width + sliceIndex * width * height
        slice[y][x] = volume.data[volumeIndex]
      }
    }

    // Normalize to [0,1]
    let min = Number.POSITIVE_INFINITY
    let max = Number.NEGATIVE_INFINITY

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        min = Math.min(min, slice[y][x])
        max = Math.max(max, slice[y][x])
      }
    }

    const range = max - min
    if (range > 0) {
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          slice[y][x] = (slice[y][x] - min) / range
        }
      }
    }

    return slice
  }

  return (
      <Card className="w-full">
        <CardContent className="pt-6">
          <h2 className="text-xl font-bold mb-4">Enhanced Model Training</h2>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Training Data</label>
              <div className="text-sm text-gray-500">
                {trainingVolumes.length} volumes available for training
                {trainingVolumes.length > 0 && (
                    <ul className="mt-1 list-disc list-inside">
                      <li>{trainingVolumes.filter((v) => v.groundTruth).length} with ground truth</li>
                      <li>
                        {trainingVolumes.length - trainingVolumes.filter((v) => v.groundTruth).length} without ground truth
                      </li>
                    </ul>
                )}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Validation Data</label>
              <div className="text-sm text-gray-500">
                {validationVolumes.length} volumes available for validation
                {validationVolumes.length > 0 && (
                    <ul className="mt-1 list-disc list-inside">
                      <li>{testingSubjects} for testing during training</li>
                      <li>{validationSubjects} for final validation</li>
                    </ul>
                )}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Current Step: {currentStep}</label>
              <div className="text-sm text-gray-500">
                {currentStep === 1 && "Brain segmentation (binary)"}
                {currentStep === 2 && "White matter segmentation (binary)"}
                {currentStep === 3 && "Multi-class segmentation (4 classes)"}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Epochs: {totalEpochs}</label>
              <Slider
                  value={[totalEpochs]}
                  min={10}
                  max={200}
                  step={10}
                  onValueChange={(value) => setTotalEpochs(value[0])}
                  disabled={isTraining}
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Early Stopping Patience: {patience}</label>
              <Slider
                  value={[patience]}
                  min={5}
                  max={30}
                  step={1}
                  onValueChange={(value) => setPatience(value[0])}
                  disabled={isTraining}
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Save Interval: {saveInterval} epochs</label>
              <Slider
                  value={[saveInterval]}
                  min={1}
                  max={50}
                  step={1}
                  onValueChange={(value) => setSaveInterval(value[0])}
                  disabled={isTraining}
              />
              <div className="flex items-center space-x-2 mt-2">
                <Checkbox
                    id="stop-on-save"
                    checked={stopOnSave}
                    onCheckedChange={(checked) => setStopOnSave(!!checked)}
                    disabled={isTraining}
                />
                <label htmlFor="stop-on-save" className="text-sm">
                  Stop training after saving model
                </label>
              </div>
            </div>

            <div className="flex gap-2">
              {!isTraining ? (
                  <Button
                      onClick={startTraining}
                      disabled={
                          !tfReady || trainingVolumes.length === 0 || trainingVolumes.filter((v) => v.groundTruth).length === 0
                      }
                      className="flex-1"
                  >
                    {tfReady ? `Train Step ${currentStep}` : "Initializing TensorFlow.js..."}
                  </Button>
              ) : (
                  <Button onClick={stopTraining} variant="destructive" className="flex-1">
                    Stop Training
                  </Button>
              )}

              <Button onClick={downloadModel} disabled={!modelTrained || isTraining} variant="outline" className="flex-1">
                Download Model
              </Button>
            </div>

            {isTraining && (
                <div className="mt-4">
                  <div className="flex justify-between text-sm mb-1">
                    <span>Progress</span>
                    <span>
                  {epoch}/{totalEpochs} epochs
                </span>
                  </div>
                  <Progress value={progress} className="h-2" />
                </div>
            )}

            {/* Display prediction results */}
            {predictionResults && (
                <div className="mt-4 p-4 bg-gray-50 rounded-md border">
                  <h3 className="text-lg font-medium mb-2">Step {predictionResults.step} Results</h3>
                  <div className="flex flex-col items-center">
                    {predictionResults.image && (
                        <div className="mb-2">
                          <img
                              src={predictionResults.image || "/placeholder.svg"}
                              alt={`Step ${predictionResults.step} prediction`}
                              className="border border-gray-300 rounded-md"
                              width={128}
                              height={128}
                          />
                        </div>
                    )}
                    {predictionResults.dice !== null && (
                        <div className="text-sm">
                          <span className="font-medium">Dice Coefficient: </span>
                          <span
                              className={`${predictionResults.dice > 0.8 ? "text-green-600" : predictionResults.dice > 0.6 ? "text-yellow-600" : "text-red-600"}`}
                          >
                      {predictionResults.dice.toFixed(4)}
                    </span>
                        </div>
                    )}
                  </div>
                </div>
            )}

            {trainingLogs.length > 0 && (
                <div className="mt-4">
                  <h3 className="text-sm font-medium mb-2">Training Logs</h3>
                  <div className="bg-gray-100 p-2 rounded-md h-40 overflow-y-auto text-xs font-mono">
                    {trainingLogs.map((log, index) => (
                        <div key={index} className="mb-1">
                          {log}
                        </div>
                    ))}
                  </div>
                </div>
            )}
            {stepCompleted && (
                <>
                  <StepPredictionViewer
                      step={currentStep}
                      t1Volume={trainingVolumes.length > 0 ? trainingVolumes[0].t1 : null}
                      t2Volume={trainingVolumes.length > 0 ? trainingVolumes[0].t2 : null}
                      groundTruthVolume={trainingVolumes.length > 0 ? trainingVolumes[0].groundTruth || null : null}
                      unetModel={unetRef.current}
                  />

                  {currentStep < 3 && (
                      <div className="mt-4">
                        <Button onClick={advanceToNextStep} className="w-full">
                          Continue to Step {currentStep + 1}
                        </Button>
                      </div>
                  )}
                </>
            )}
          </div>
        </CardContent>
      </Card>
  )
}
