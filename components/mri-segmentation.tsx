"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { InfoIcon } from "lucide-react"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import AnalyzeFileUploader from "./analyze-file-uploader"
import VolumeViewer from "./volume-viewer"
import { EnhancedUNet } from "@/lib/unet-enhanced"
import ModelTrainerEnhanced from "./model-trainer-enhanced"
import PostProcessing from "./post-processing"
import EvaluationMetrics from "./evaluation-metrics"
import DebugInfo from "./debug-info"
import ModelManager from "./model-manager"
import type { AnalyzeVolume } from "@/lib/analyze-parser"
import * as tf from "@tensorflow/tfjs"
// Add the import for BatchFileUploader
import BatchFileUploader from "./batch-file-uploader"
// Add the import for DatasetManager
import DatasetManager from "./dataset-manager"

// Add the Subject interface after the existing imports
interface Subject {
  id: string
  t1Volume?: AnalyzeVolume
  t2Volume?: AnalyzeVolume
  labelVolume?: AnalyzeVolume
  files: any
  isComplete: boolean
  isLoaded: boolean
}

export default function MriSegmentation() {
  const [t1Volume, setT1Volume] = useState<AnalyzeVolume | null>(null)
  const [t2Volume, setT2Volume] = useState<AnalyzeVolume | null>(null)
  const [groundTruthVolume, setGroundTruthVolume] = useState<AnalyzeVolume | null>(null)
  const [segmentationVolume, setSegmentationVolume] = useState<AnalyzeVolume | null>(null)
  const [processedSegmentationVolume, setProcessedSegmentationVolume] = useState<AnalyzeVolume | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [activeTab, setActiveTab] = useState("upload")
  const unetRef = useRef<EnhancedUNet | null>(null)
  // Add a state to track if a model is loaded
  const [modelLoaded, setModelLoaded] = useState(false)
  const [modelLoadError, setModelLoadError] = useState<string | null>(null)
  const [currentModelPath, setCurrentModelPath] = useState<string | null>("indexeddb://unet-iseg-model")

  // Add states for modality selection
  const [useT1, setUseT1] = useState(true)
  const [useT2, setUseT2] = useState(true)
  const [useGroundTruth, setUseGroundTruth] = useState(false)

  // Training and validation data
  const [trainingVolumes, setTrainingVolumes] = useState<
      { t1: AnalyzeVolume; t2: AnalyzeVolume; groundTruth?: AnalyzeVolume }[]
  >([])
  const [validationVolumes, setValidationVolumes] = useState<
      { t1: AnalyzeVolume; t2: AnalyzeVolume; groundTruth?: AnalyzeVolume }[]
  >([])

  useEffect(() => {
    const initModel = async () => {
      try {
        // Initialize TensorFlow.js
        await tf.ready()
        console.log("TensorFlow.js ready")

        // Initialize U-Net model
        unetRef.current = new EnhancedUNet(currentModelPath || undefined)
        await unetRef.current.initialize()
        console.log("Enhanced U-Net model initialized")

        // Check if a model is already saved
        try {
          const modelInfo = await tf.io.listModels()
          if (modelInfo[currentModelPath || "indexeddb://unet-iseg-model"]) {
            setModelLoaded(true)
            console.log("Found previously trained model in storage")
          }
        } catch (e) {
          console.log("No previously trained model found")
        }
      } catch (error) {
        console.error("Error initializing model:", error)
        setModelLoadError(`Error initializing model: ${error instanceof Error ? error.message : String(error)}`)
      }
    }

    initModel()
  }, [currentModelPath])

  const handleSegmentation = async () => {
    // Check if we have the required volumes based on selection
    if ((useT1 && !t1Volume) || (useT2 && !t2Volume) || (useGroundTruth && !groundTruthVolume)) {
      alert("Please provide all selected input volumes")
      return
    }

    // Make sure at least one modality is selected
    if (!useT1 && !useT2 && !useGroundTruth) {
      alert("Please select at least one input modality")
      return
    }

    setIsProcessing(true)
    setProgress(0)

    try {
      // If we're just viewing volumes without segmentation (no model needed)
      if (!modelLoaded && useGroundTruth && groundTruthVolume) {
        console.log("Viewing ground truth without segmentation")
        setProgress(50)

        // Just use the ground truth as the segmentation result
        setSegmentationVolume(groundTruthVolume)
        setProcessedSegmentationVolume(groundTruthVolume)

        setProgress(100)
        setActiveTab("results")
        setIsProcessing(false)
        return
      }

      // Regular segmentation with model
      if (!modelLoaded) {
        setProgress(10)
        try {
          // Try to load the model from IndexedDB
          await unetRef.current?.initialize()
          setModelLoaded(true)
          setProgress(20)
        } catch (error) {
          throw new Error("No trained model found. Please train the model first.")
        }
      } else {
        setProgress(20)
      }

      // Collect the selected input volumes
      const inputVolumes: tf.Tensor3D[] = []

      // Preprocess volumes
      setProgress(30)
      if (useT1 && t1Volume) {
        console.log("Preprocessing T1 volume...")
        const t1Tensor = volumeToTensor(t1Volume)
        console.log("T1 tensor shape:", t1Tensor.shape)
        inputVolumes.push(t1Tensor)
      }

      setProgress(40)
      if (useT2 && t2Volume) {
        console.log("Preprocessing T2 volume...")
        const t2Tensor = volumeToTensor(t2Volume)
        console.log("T2 tensor shape:", t2Tensor.shape)
        inputVolumes.push(t2Tensor)
      }

      setProgress(50)
      if (useGroundTruth && groundTruthVolume) {
        console.log("Preprocessing Ground Truth volume...")
        const gtTensor = volumeToTensor(groundTruthVolume)
        console.log("Ground Truth tensor shape:", gtTensor.shape)
        inputVolumes.push(gtTensor)
      }

      // Check if dimensions match
      for (let i = 1; i < inputVolumes.length; i++) {
        if (!tf.util.arraysEqual(inputVolumes[0].shape, inputVolumes[i].shape)) {
          throw new Error(
              `Input volumes must have the same dimensions. Volume 0: ${inputVolumes[0].shape}, Volume ${i}: ${inputVolumes[i].shape}`,
          )
        }
      }

      // Initialize model with the correct number of input channels
      await unetRef.current?.initialize(inputVolumes.length)

      setProgress(60)
      console.log("Running segmentation with trained model...")

      // Run segmentation with post-processing
      const result = await unetRef.current?.predictWithPostProcessing(inputVolumes)
      if (!result) {
        throw new Error("Segmentation failed: No result returned from model")
      }
      console.log("Segmentation result shape:", result.shape)

      // Convert result tensor to volume
      const segVolume = tensorToVolume(result, t1Volume || t2Volume || groundTruthVolume!)
      setSegmentationVolume(segVolume)

      setProgress(80)
      console.log("Segmentation complete")

      // Apply post-processing
      console.log("Applying post-processing...")
      // Post-processing is already applied in predictWithPostProcessing
      setProcessedSegmentationVolume(segVolume)

      setProgress(100)
      setActiveTab("results")
    } catch (error) {
      console.error("Segmentation failed:", error)
      // Display error to user
      alert(`Segmentation failed: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setIsProcessing(false)
    }
  }

  const resetVolumes = () => {
    setT1Volume(null)
    setT2Volume(null)
    setGroundTruthVolume(null)
    setSegmentationVolume(null)
    setProcessedSegmentationVolume(null)
    setActiveTab("upload")
  }

  // Helper function to convert volume to tensor
  const volumeToTensor = (volume: AnalyzeVolume): tf.Tensor3D => {
    const { dimensions, dataTypeString } = volume.header
    const [width, height, depth] = dimensions

    // Create a new tensor with the volume data
    const tensor = tf.tensor3d(Array.from(volume.data), [width, height, depth])

    // Normalize to [0, 1]
    const normalized = tf.div(tf.sub(tensor, tf.scalar(volume.min)), tf.scalar(volume.max - volume.min))

    return normalized
  }

  // Helper function to convert tensor to volume
  const tensorToVolume = (tensor: tf.Tensor3D, headerTemplate: any): AnalyzeVolume => {
    // Get tensor data
    const data = tensor.dataSync()

    // Create a new volume with the tensor data
    return {
      header: { ...headerTemplate.header },
      data: Float32Array.from(data),
      min: 0,
      max: tensor.max().dataSync()[0],
    }
  }

  // Add volume to training data
  const addToTrainingData = () => {
    if (t1Volume && t2Volume) {
      setTrainingVolumes((prev) => [
        ...prev,
        {
          t1: t1Volume,
          t2: t2Volume,
          groundTruth: groundTruthVolume || undefined,
        },
      ])
      alert("Current volumes added to training data")
    }
  }

  // Add volume to validation data
  const addToValidationData = () => {
    if (t1Volume && t2Volume) {
      setValidationVolumes((prev) => [
        ...prev,
        {
          t1: t1Volume,
          t2: t2Volume,
          groundTruth: groundTruthVolume || undefined,
        },
      ])
      alert("Current volumes added to validation data")
    }
  }

  // Add the handleSubjectsLoaded function inside the MriSegmentation component, before the return statement
  const handleSubjectsLoaded = (subjects: Subject[], forTraining: boolean) => {
    // Filter out subjects that don't have both T1 and T2 volumes
    const validSubjects = subjects.filter((subject) => subject.t1Volume && subject.t2Volume)

    if (validSubjects.length === 0) {
      alert("No valid subjects found with both T1 and T2 volumes")
      return
    }

    // Convert subjects to the format expected by the training/validation data
    const volumeData = validSubjects.map((subject) => ({
      t1: subject.t1Volume!,
      t2: subject.t2Volume!,
      groundTruth: subject.labelVolume,
    }))

    if (forTraining) {
      setTrainingVolumes((prev) => [...prev, ...volumeData])
      alert(`Added ${validSubjects.length} subjects to training data`)
    } else {
      setValidationVolumes((prev) => [...prev, ...volumeData])
      alert(`Added ${validSubjects.length} subjects to validation data`)
    }

    // If we have at least one subject, set the current volumes to the first one
    if (validSubjects.length > 0) {
      const firstSubject = validSubjects[0]
      setT1Volume(firstSubject.t1Volume!)
      setT2Volume(firstSubject.t2Volume!)
      if (firstSubject.labelVolume) {
        setGroundTruthVolume(firstSubject.labelVolume)
      }
    }
  }

  // Add these functions inside the MriSegmentation component, before the return statement
  const removeTrainingItem = (index: number) => {
    setTrainingVolumes((prev) => prev.filter((_, i) => i !== index))
  }

  const removeValidationItem = (index: number) => {
    setValidationVolumes((prev) => prev.filter((_, i) => i !== index))
  }

  const clearTrainingData = () => {
    if (confirm("Are you sure you want to clear all training data?")) {
      setTrainingVolumes([])
    }
  }

  const clearValidationData = () => {
    if (confirm("Are you sure you want to clear all validation data?")) {
      setValidationVolumes([])
    }
  }

  const loadDatasetItem = (item: { t1: AnalyzeVolume; t2: AnalyzeVolume; groundTruth?: AnalyzeVolume }) => {
    setT1Volume(item.t1)
    setT2Volume(item.t2)
    if (item.groundTruth) {
      setGroundTruthVolume(item.groundTruth)
    } else {
      setGroundTruthVolume(null)
    }
  }

  // Add a function to automatically split data into training, testing, and validation sets
  // Add this function before the return statement
  const splitDatasets = () => {
    // Get all volumes that have both T1 and T2
    const allVolumes = [...trainingVolumes, ...validationVolumes].filter((vol) => vol.t1 && vol.t2 && vol.groundTruth)

    if (allVolumes.length < 10) {
      alert(`Need at least 10 subjects for 6-2-2 split. Currently have ${allVolumes.length}.`)
      return
    }

    // Shuffle the volumes
    const shuffled = [...allVolumes].sort(() => 0.5 - Math.random())

    // Split into 6-2-2
    const trainSet = shuffled.slice(0, 6)
    const testSet = shuffled.slice(6, 8)
    const validSet = shuffled.slice(8, 10)

    // Update the state
    setTrainingVolumes(trainSet)
    setValidationVolumes([...testSet, ...validSet])

    alert(`Data split complete: 6 for training, 2 for testing, 2 for validation`)
  }

  // Add a function to handle model training completion
  const handleModelTrained = () => {
    setModelLoaded(true)
    setActiveTab("upload")
  }

  // Add a function to handle custom model loading
  const handleModelLoaded = (modelPath: string) => {
    setCurrentModelPath(modelPath)
    setModelLoaded(false) // Will be set to true when model is initialized in useEffect
  }

  return (
      <div className="w-full max-w-6xl">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          {/* Update the tabs to make the workflow clearer */}
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="upload">1. Upload Data</TabsTrigger>
            <TabsTrigger value="train" disabled={trainingVolumes.length === 0}>
              2. Train Model
            </TabsTrigger>
            <TabsTrigger value="results" disabled={!segmentationVolume}>
              3. Results & Evaluation
            </TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="space-y-6">
            {/* Add model status alert */}
            {modelLoaded ? (
                <Alert className="bg-green-50 border-green-200">
                  <InfoIcon className="h-4 w-4 text-green-600" />
                  <AlertTitle>Model Loaded</AlertTitle>
                  <AlertDescription>
                    A trained model is loaded and ready for segmentation. You can run segmentation on new data or continue
                    training.
                  </AlertDescription>
                </Alert>
            ) : modelLoadError ? (
                <Alert className="bg-red-50 border-red-200">
                  <InfoIcon className="h-4 w-4 text-red-600" />
                  <AlertTitle>Model Error</AlertTitle>
                  <AlertDescription>{modelLoadError}</AlertDescription>
                </Alert>
            ) : (
                <Alert>
                  <InfoIcon className="h-4 w-4" />
                  <AlertTitle>No Model Loaded</AlertTitle>
                  <AlertDescription>
                    No trained model is currently loaded. Please train a model before running segmentation.
                  </AlertDescription>
                </Alert>
            )}

            {/* Add the ModelManager component */}
            <ModelManager onModelLoaded={handleModelLoaded} currentModelPath={currentModelPath} />

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card>
                <CardContent className="pt-6">
                  <h3 className="text-lg font-medium mb-4">T1 Volume</h3>
                  <AnalyzeFileUploader onVolumeLoaded={setT1Volume} volumeType="T1" currentVolume={t1Volume} />
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <h3 className="text-lg font-medium mb-4">T2 Volume</h3>
                  <AnalyzeFileUploader onVolumeLoaded={setT2Volume} volumeType="T2" currentVolume={t2Volume} />
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <h3 className="text-lg font-medium mb-4">Ground Truth (Optional)</h3>
                  <AnalyzeFileUploader
                      onVolumeLoaded={setGroundTruthVolume}
                      volumeType="Ground Truth"
                      currentVolume={groundTruthVolume}
                      optional={true}
                  />
                </CardContent>
              </Card>
            </div>

            {/* Add modality selection */}
            <Card>
              <CardContent className="pt-6">
                <h3 className="text-lg font-medium mb-4">Segmentation Options</h3>
                <div className="space-y-4">
                  <div className="flex items-center space-x-2">
                    <Checkbox id="use-t1" checked={useT1} onCheckedChange={(checked) => setUseT1(!!checked)} />
                    <Label htmlFor="use-t1">Use T1 Volume</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox id="use-t2" checked={useT2} onCheckedChange={(checked) => setUseT2(!!checked)} />
                    <Label htmlFor="use-t2">Use T2 Volume</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox
                        id="use-gt"
                        checked={useGroundTruth}
                        onCheckedChange={(checked) => setUseGroundTruth(!!checked)}
                        disabled={!groundTruthVolume}
                    />
                    <Label htmlFor="use-gt" className={!groundTruthVolume ? "text-gray-400" : ""}>
                      Use Ground Truth Volume (for guided segmentation)
                    </Label>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Add the BatchFileUploader component to the upload tab, right after the existing file uploaders */}
            <BatchFileUploader onSubjectsLoaded={handleSubjectsLoaded} />

            {/* Add the DatasetManager component to the upload tab, after the BatchFileUploader */}
            <DatasetManager
                trainingData={trainingVolumes}
                validationData={validationVolumes}
                onRemoveTrainingItem={removeTrainingItem}
                onRemoveValidationItem={removeValidationItem}
                onClearTrainingData={clearTrainingData}
                onClearValidationData={clearValidationData}
                onLoadItem={loadDatasetItem}
            />

            <div className="flex flex-wrap justify-center gap-4 mt-8">
              <Button
                  onClick={handleSegmentation}
                  disabled={
                      isProcessing ||
                      (!useT1 && !useT2 && !useGroundTruth) ||
                      (useT1 && !t1Volume) ||
                      (useT2 && !t2Volume) ||
                      (useGroundTruth && !groundTruthVolume) ||
                      (!modelLoaded && !useGroundTruth)
                  }
                  className="w-48"
                  variant={modelLoaded ? "default" : "outline"}
              >
                {isProcessing ? "Processing..." : "Run Segmentation"}
              </Button>

              <Button onClick={addToTrainingData} disabled={!t1Volume || !t2Volume} variant="outline" className="w-48">
                Add to Training Data
              </Button>

              <Button onClick={addToValidationData} disabled={!t1Volume || !t2Volume} variant="outline" className="w-48">
                Add to Validation Data
              </Button>

              {/* Add a button for automatic data splitting in the upload tab */}
              <Button
                  onClick={splitDatasets}
                  disabled={trainingVolumes.length + validationVolumes.length < 10}
                  variant="outline"
                  className="w-48"
              >
                Auto-Split (6-2-2)
              </Button>
            </div>

            {isProcessing && (
                <div className="mt-4">
                  <Progress value={progress} className="h-2" />
                  <p className="text-center mt-2 text-sm text-gray-500">Processing: {progress}%</p>
                </div>
            )}

            <DebugInfo t1Volume={t1Volume} t2Volume={t2Volume} groundTruthVolume={groundTruthVolume} />

            <Card>
              <CardContent className="pt-6">
                <h3 className="text-lg font-medium mb-4">Dataset Status</h3>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Training Volumes:</span>
                    <span className="font-bold">{trainingVolumes.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Validation Volumes:</span>
                    <span className="font-bold">{validationVolumes.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Model Status:</span>
                    <span className={`font-bold ${modelLoaded ? "text-green-600" : "text-red-600"}`}>
                    {modelLoaded ? "Trained" : "Not Trained"}
                  </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Model Path:</span>
                    <span className="font-bold text-sm truncate max-w-[200px]">
                    {currentModelPath?.startsWith("indexeddb://custom")
                        ? "Custom Uploaded Model"
                        : currentModelPath || "Default Model"}
                  </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="train" className="space-y-6">
            <ModelTrainerEnhanced
                onModelTrained={handleModelTrained}
                trainingVolumes={trainingVolumes}
                validationVolumes={validationVolumes}
            />
          </TabsContent>

          <TabsContent value="results" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {t1Volume && useT1 && (
                  <Card>
                    <CardContent className="pt-6">
                      <h3 className="text-lg font-medium mb-4">T1 Volume</h3>
                      <VolumeViewer volume={t1Volume} />
                    </CardContent>
                  </Card>
              )}

              {t2Volume && useT2 && (
                  <Card>
                    <CardContent className="pt-6">
                      <h3 className="text-lg font-medium mb-4">T2 Volume</h3>
                      <VolumeViewer volume={t2Volume} />
                    </CardContent>
                  </Card>
              )}

              {groundTruthVolume && useGroundTruth && (
                  <Card>
                    <CardContent className="pt-6">
                      <h3 className="text-lg font-medium mb-4">Ground Truth Input</h3>
                      <VolumeViewer volume={groundTruthVolume} isSegmentation={true} />
                    </CardContent>
                  </Card>
              )}

              {segmentationVolume && (
                  <Card>
                    <CardContent className="pt-6">
                      <h3 className="text-lg font-medium mb-4">Segmentation Result</h3>
                      <VolumeViewer volume={segmentationVolume} isSegmentation={true} />
                    </CardContent>
                  </Card>
              )}
            </div>

            {segmentationVolume && t1Volume && (
                <Card>
                  <CardContent className="pt-6">
                    <h3 className="text-lg font-medium mb-4">Segmentation with T1 Overlay</h3>
                    <VolumeViewer volume={t1Volume} overlayVolume={segmentationVolume} />
                  </CardContent>
                </Card>
            )}

            {/* Add post-processing component */}
            {segmentationVolume && (
                <PostProcessing
                    segmentationVolume={segmentationVolume}
                    groundTruthVolume={groundTruthVolume}
                    t1Volume={t1Volume}
                />
            )}

            {/* Add evaluation metrics component */}
            {segmentationVolume && groundTruthVolume && (
                <EvaluationMetrics segmentationVolume={segmentationVolume} groundTruthVolume={groundTruthVolume} />
            )}

            <div className="flex justify-center mt-8">
              <Button onClick={resetVolumes} variant="outline" className="w-48">
                Reset Volumes
              </Button>
            </div>
          </TabsContent>
        </Tabs>
      </div>
  )
}
