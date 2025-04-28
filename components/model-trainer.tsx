"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Slider } from "@/components/ui/slider"
import { UNet } from "@/lib/unet"
import * as tf from "@tensorflow/tfjs"

interface ModelTrainerProps {
  onModelTrained: () => void
}

export default function ModelTrainer({ onModelTrained }: ModelTrainerProps) {
  const [isTraining, setIsTraining] = useState(false)
  const [progress, setProgress] = useState(0)
  const [epoch, setEpoch] = useState(0)
  const [totalEpochs, setTotalEpochs] = useState(10)
  const [batchSize, setBatchSize] = useState(4)
  const [learningRate, setLearningRate] = useState(0.0001)
  const [trainingLogs, setTrainingLogs] = useState<string[]>([])
  const unetRef = useRef<UNet | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [datasetSize, setDatasetSize] = useState(0)

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files) return

    setTrainingLogs((prev) => [...prev, `Loading ${files.length} files...`])
    setDatasetSize(Math.floor(files.length / 3)) // Assuming each subject has T1, T2, and label
  }

  const startTraining = async () => {
    if (!unetRef.current) {
      unetRef.current = new UNet()
      await unetRef.current.initialize()
    }

    setIsTraining(true)
    setProgress(0)
    setEpoch(0)
    setTrainingLogs((prev) => [...prev, "Starting training..."])

    try {
      // In a real implementation, we would load the dataset here
      // For this demo, we'll simulate training

      const optimizer = tf.train.adam(learningRate)

      // Create a callback to track progress
      const callbacks = {
        onEpochEnd: (epoch: number, logs: any) => {
          setEpoch(epoch + 1)
          setProgress(((epoch + 1) / totalEpochs) * 100)
          setTrainingLogs((prev) => [
            ...prev,
            `Epoch ${epoch + 1}/${totalEpochs} - loss: ${logs.loss.toFixed(4)} - accuracy: ${logs.acc.toFixed(4)}`,
          ])
        },
      }

      // Simulate training
      for (let i = 0; i < totalEpochs; i++) {
        await new Promise((resolve) => setTimeout(resolve, 1000))
        callbacks.onEpochEnd(i, {
          loss: Math.random() * 0.5,
          acc: 0.7 + Math.random() * 0.2,
        })
      }

      setTrainingLogs((prev) => [...prev, "Training complete!"])
      onModelTrained()
    } catch (error) {
      console.error("Training failed:", error)
      setTrainingLogs((prev) => [...prev, `Error: ${error}`])
    } finally {
      setIsTraining(false)
    }
  }

  return (
    <Card className="w-full">
      <CardContent className="pt-6">
        <h2 className="text-xl font-bold mb-4">Model Training</h2>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Upload Training Dataset</label>
            <div className="flex items-center gap-2">
              <Button onClick={() => fileInputRef.current?.click()} variant="outline" disabled={isTraining}>
                Select Files
              </Button>
              <span className="text-sm text-gray-500">
                {datasetSize > 0 ? `${datasetSize} subjects loaded` : "No files selected"}
              </span>
            </div>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              multiple
              accept="image/*"
              className="hidden"
            />
            <p className="text-xs text-gray-500 mt-1">Upload T1, T2, and label images for each subject</p>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Epochs: {totalEpochs}</label>
            <Slider
              value={[totalEpochs]}
              min={1}
              max={50}
              step={1}
              onValueChange={(value) => setTotalEpochs(value[0])}
              disabled={isTraining}
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Batch Size: {batchSize}</label>
            <Slider
              value={[batchSize]}
              min={1}
              max={16}
              step={1}
              onValueChange={(value) => setBatchSize(value[0])}
              disabled={isTraining}
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Learning Rate: {learningRate}</label>
            <Slider
              value={[learningRate * 10000]}
              min={1}
              max={100}
              step={1}
              onValueChange={(value) => setLearningRate(value[0] / 10000)}
              disabled={isTraining}
            />
          </div>

          <Button onClick={startTraining} disabled={isTraining || datasetSize === 0} className="w-full">
            {isTraining ? "Training..." : "Start Training"}
          </Button>

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
        </div>
      </CardContent>
    </Card>
  )
}
